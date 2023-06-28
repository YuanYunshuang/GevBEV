import logging, os
from typing import Tuple, Any

import torch, torch_scatter
import torch.nn.functional as F
import numpy as np
from torch import Tensor

from utils import pclib, box_utils, vislib
from ops.utils import points_in_boxes_gpu
from model.submodules.utils import meshgrid, metric2indices, draw_sample_prob, pad_l, pad_r
import matplotlib.pyplot as plt


class Compose(object):
    """Composes several pre-processing modules together.
        Take care that these functions modify the input data directly.
    """

    def __init__(self, processes):
        self.processes = processes

    def __call__(self, *args):
        out = {}  # for post processors
        for t in self.processes:
            t(*args)
            if hasattr(t, 'out'):
                out.update(t.out)
        return out

    def set_vis_dir(self, vis_dir):
        for t in self.processes:
            t.vis_dir = vis_dir


class DimensionlessCoordinates(object):
    def __init__(self, voxel_size=0.1):
        self.voxel_size = voxel_size
        if isinstance(voxel_size, list) or getattr(voxel_size, '__len__', False):
            # list or other iterable, such as ListConfig
            self.voxel_size = np.array(voxel_size).reshape(1, -1)
        logging.info(f"{self.__class__.__name__} with voxel_size:{voxel_size}")

    def __call__(self, data_dict):
        if isinstance(data_dict['coords'], list):
            assert isinstance(self.voxel_size, float) or \
                   self.voxel_size.shape[-1] == data_dict['coords'][0].shape[-1]
            data_dict['coords'] = [coords / self.voxel_size
                                   for coords in data_dict['coords']]
        else:
            assert isinstance(self.voxel_size, float) or \
                   self.voxel_size.shape[-1] == data_dict['coords'].shape[-1]
            data_dict['coords'] = data_dict['coords'] / self.voxel_size


#####POST PROCESSORS#####
class DistributionPostProcess(object):
    def __init__(self,
                 voxel_size,
                 stride,
                 lidar_range,
                 distr_r,
                 det_r=None,
                 visualization=False,
                 vis_dir=None,
                 edl=True):
        super(DistributionPostProcess, self).__init__()
        self.lidar_range = lidar_range
        self.det_r = det_r
        self.distr_r = distr_r
        self.voxel_size = voxel_size[0] if isinstance(voxel_size, list) else voxel_size
        self.stride = stride
        self.vis = visualization
        self.vis_dir = vis_dir
        self.edl = edl

        self.n_sam_per_dim = 29
        self.unit_box_template = meshgrid(
            -1.4, 1.4, dim=2, n_steps=self.n_sam_per_dim
        ).view(1, self.n_sam_per_dim, self.n_sam_per_dim, 2).cuda()

        self.out = {}

    def __call__(self, batch_dict):
        self.out = {'frame_id': batch_dict['frame_id'],}
        if 'distr_surface' in batch_dict:
            self.surface(batch_dict)
        if 'distr_object' in batch_dict:
            self.object(batch_dict)
        if self.vis:
            self.visualization(batch_dict)

    def surface(self, batch_dict):
        res = self.voxel_size * self.stride['surface']
        grid_size = int(self.det_r / res * 2)
        evidence = batch_dict['distr_surface']['evidence']
        obs_mask = batch_dict['distr_surface']['obs_mask']

        s = batch_dict['bevmap_static'].shape[2] // grid_size
        road_bev = batch_dict['bevmap_static'][:, ::s, ::s].bool()

        conf, unc = self.evidence_to_conf_unc(evidence)

        self.out.update({
            'road_confidence': conf,
            'cared_mask': road_bev,
            'road_uncertainty': unc.squeeze(),
            'road_obs_mask': obs_mask,
            'road_Nall': batch_dict['distr_surface']['Nall'],
            'road_Nsel': batch_dict['distr_surface']['Nsel']
        })

    def object(self, batch_dict):
        pred_boxes = None
        if 'detection' in batch_dict:
            pred_boxes = batch_dict['detection']['pred_boxes']
            assert pred_boxes.shape[1] == 8
        gt_boxes = batch_dict['target_boxes']
        assert gt_boxes.shape[1] == 8
        evidence = batch_dict['distr_object']['evidence']
        obs_mask = batch_dict['distr_object']['obs_mask']

        conf, unc = self.evidence_to_conf_unc(evidence)

        pred_sam_coor, pred_box_unc, pred_box_conf = \
            self.get_sample_probs_pixel(pred_boxes, unc, conf)
        gt_sam_coor, gt_box_unc, gt_box_conf = \
            self.get_sample_probs_pixel(gt_boxes, unc, conf)
        bev_conf_p1, bev_unc_p1 = None, None

        self.out.update({
            'box_bev_unc': unc,
            'box_bev_conf': conf,
            'box_bev_unc_p1': bev_unc_p1,
            'box_bev_conf_p1': bev_conf_p1,
            'box_obs_mask': obs_mask,
            'pred_boxes': pred_boxes,
            'gt_boxes': gt_boxes,
            'pred_box_samples': pred_sam_coor,
            'pred_box_unc': pred_box_unc,
            'pred_box_conf': pred_box_conf,
            'gt_box_samples': gt_sam_coor,
            'gt_box_unc': gt_box_unc,
            'gt_box_conf': gt_box_conf,
            'box_Nall': batch_dict['distr_object']['Nall'],
            'box_Nsel': batch_dict['distr_object']['Nsel']
        })
    
    def visualization(self, batch_dict):
        points = batch_dict['xyz']
        num_cav = batch_dict['num_cav'][0]
        points_idx0 = batch_dict['in_data'].C[:, 0] < num_cav
        points = points[points_idx0].cpu().numpy()

        # we only visualize the 1st batch
        fn = '_'.join(self.out['frame_id'][0])

        road_points = None
        if 'road_confidence' in self.out:
            res = self.voxel_size * self.stride['surface']
            grid_size = (
                round((self.lidar_range[3] - self.lidar_range[0]) / res),
                round((self.lidar_range[4] - self.lidar_range[1]) / res),
            )

            road_bev = self.out['cared_mask'][0].cpu().numpy()
            confs_np = self.out['road_confidence'][0].cpu().numpy()
            obs_msk = self.out['road_obs_mask'][0].cpu().numpy()
            unc = self.out['road_uncertainty'][0].cpu().numpy()

            road_mask = np.argmax(confs_np, axis=-1).astype(bool)
            road_points = np.stack(np.where(road_mask), axis=-1)
            road_points = road_points * res + res / 2
            road_points[:, 0] += self.lidar_range[0]
            road_points[:, 1] += self.lidar_range[1]
            road_points_prob = confs_np[road_mask, 1]

            valid = np.logical_and((unc < 1.0).squeeze(), obs_msk)
            pos_road = np.logical_and(road_mask == 1, valid)
            tp_road = np.logical_and(pos_road, road_bev.astype(bool)).sum()
            fn_road = np.logical_and(np.logical_not(pos_road),
                                     road_bev.astype(bool)).sum()
            iou_road = tp_road / (pos_road.sum() + fn_road) * 100

            bevmap = np.zeros((grid_size, grid_size, 3))
            bevmap[..., 2] = 255 * confs_np[..., 0]
            bevmap[..., 1] = 255 * confs_np[..., 1]
            gt_bevmap = np.zeros((grid_size, grid_size, 3))
            gt_bevmap[..., 1] = road_bev * 255

            fig = plt.figure(figsize=(13, 6))
            axs = fig.subplots(1, 3)
            axs[0].imshow(1 - unc.T[::-1], cmap='hot')
            axs[1].imshow(bevmap.astype(np.uint8).transpose(1, 0, 2)[::-1])
            axs[2].imshow(gt_bevmap.astype(np.uint8).transpose(1, 0, 2)[::-1])
            fig.suptitle(f"road: {iou_road:.2f}", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, fn) + '_2.png')
            plt.close()

        if 'pred_box_conf' in self.out:
            pred_boxes = self.out['pred_boxes']
            pred_b0 = pred_boxes[:, 0] == 0
            pred_boxes = pred_boxes[pred_b0, 1:].cpu().numpy()
            gt_boxes = self.out['gt_boxes']
            gt_b0 = gt_boxes[:, 0] == 0
            gt_boxes = gt_boxes[gt_b0, 1:].cpu().numpy()
            pred_box_sam = self.out['pred_box_samples'][pred_b0].cpu().numpy()
            pred_box_unc = self.out['pred_box_unc'][pred_b0].cpu().numpy()
            pred_box_conf = self.out['pred_box_conf'][pred_b0].cpu().numpy()
            gt_box_unc = self.out['gt_box_unc'][gt_b0].cpu().numpy()
            gt_box_sam = self.out['gt_box_samples'][gt_b0].cpu().numpy()
            gt_box_conf = self.out['gt_box_conf'][gt_b0].cpu().numpy()

            # draw road uncertainty (if exists), lidar points, pred & gt box, pred & gt boxe distribution
            fig = plt.figure(figsize=((self.lidar_range[3] - self.lidar_range[0]) / 10,
                                      (self.lidar_range[4] - self.lidar_range[1]) / 10))
            ax = fig.subplots()
            ax.set_facecolor((0.0, 0.0, 0.0))
            if road_points is not None:
                ax.scatter(road_points[:, 0], road_points[:, 1],
                           c=road_points_prob, cmap='hot', s=1, vmin=0, vmax=1)
            ax.plot(points[:, 0], points[:, 1], '.', markersize=0.5)

            s = 4
            for i, (sam, conf) in enumerate(zip(pred_box_sam, pred_box_conf)):
                ax.scatter(sam[s:-s, s:-s, 0], sam[s:-s, s:-s, 1], c=conf[s:-s, s:-s, 1],
                           cmap='cool', s=.5, vmin=0, vmax=1)

                # ax.contour(sam[..., 0], sam[..., 1], conf[..., 1],
                #            cmap='cool', vmin=0, vmax=1)
            for i, (sam, conf) in enumerate(zip(gt_box_sam, gt_box_conf)):
                ax.scatter(sam[s:-s, s:-s, 0], sam[s:-s, s:-s, 1], c=conf[s:-s, s:-s, 1],
                           cmap='cool', s=.5, vmin=0, vmax=1)
                # ax.contour(sam[..., 0], sam[..., 1], conf[..., 1],
                #            cmap='cool', vmin=0, vmax=1)

            ax = vislib.draw_box_plt(pred_boxes, ax, color='r')
            ax = vislib.draw_box_plt(gt_boxes, ax, color='g')

            ax.set_xlim([self.lidar_range[0], self.lidar_range[3]])
            ax.set_ylim([self.lidar_range[1], self.lidar_range[4]])
            plt.tight_layout()
            plt.savefig(os.path.join(self.vis_dir, fn) + '_1.png')
            plt.close()

            # draw box bev map
            box_bev_conf = self.out['box_bev_conf'][0, :, :, 1].cpu().numpy().T
            plt.imshow(box_bev_conf[::-1], cmap='jet', vmin=0, vmax=1)
            plt.savefig(os.path.join(self.vis_dir, fn) + '_0.png')
            plt.close()

    def evidence_to_conf_unc(self, evidence):
        if self.edl:
        # used edl loss
            alpha = evidence + 1
            S = torch.sum(alpha, dim=-1, keepdim=True)
            conf = torch.div(alpha, S)
            K = evidence.shape[-1]
            unc = torch.div(K, S)
            # conf = torch.sqrt(conf * (1 - unc))
            unc = unc.squeeze(dim=-1)
        else:
            # use entropy as uncertainty
            entropy = -evidence * torch.log2(evidence)
            unc = entropy.sum(dim=-1)
            # conf = torch.sqrt(evidence * (1 - unc.unsqueeze(-1)))
            conf = evidence
        return conf, unc

    def get_sample_probs_metirc(self, boxes, ctrs, ctrs_reg):
        """resample points from box parameters and draw probs from box prob map"""
        res = self.voxel_size * self.stride['object']
        grid_size = (
            round((self.lidar_range[3] - self.lidar_range[0]) / res),
            round((self.lidar_range[4] - self.lidar_range[1]) / res),
        )
        # transform to lidar frame
        st = torch.sin(boxes[:, -1]).view(-1, 1, 1)
        ct = torch.cos(boxes[:, -1]).view(-1, 1, 1)
        samples = self.unit_box_template * boxes[:, 4:6].view(-1, 1, 1, 2) * 0.5
        samples = torch.stack([
            samples[..., 0] * ct - samples[..., 1] * st,
            samples[..., 0] * st + samples[..., 1] * ct
        ], dim=-1)
        samples = samples + boxes[:, 1:3].view(-1, 1, 1, 2)  # n 21 21 2
        # samples = samples.view(samples.shape[0], -1, 2)  # n 441 2

        nbox = len(boxes)
        s = self.n_sam_per_dim
        batch_indices = torch.tile(boxes[:, 0].view(-1, 1, 1), (1, s, s)).long()
        samples = torch.cat([batch_indices.unsqueeze(-1), samples], dim=-1)
        samidx = metric2indices(samples.view(-1, 3), res).view(nbox, s, s, 3)
        x = samidx[..., 1] + grid_size[0]
        y = samidx[..., 2] + grid_size[1]
        mask = (x >= 0) & (x < grid_size[0]) & (y >= 0) & (y < grid_size[1])

        evis = draw_sample_prob(ctrs, ctrs_reg, samples[mask],
                                 res, self.distr_r['object'], self.lidar_range,
                                 ctrs[:, 0].max().long().item() + 1,
                                 var0=[0.1, 0.1])
        sam_evis = torch.zeros_like(samples[..., :2])
        sam_evis[mask] = evis
        sam_conf, sam_unc = self.evidence_to_conf_unc(sam_evis)
        return samples[..., 1:], sam_unc, sam_conf

    def get_sample_probs_pixel(self, boxes, unc, conf):
        """
        Given bounding boxes, retrieve uncertainties and confidences of the sampled template points in each box
        from the uncertainty and confidence maps.

        :param boxes: (N, 8), [batch_idx, x, y, z, l, w, h, r]
        :param unc: (B, W, H, C), C indicates number of class
        :param conf: (B, W, H)
        :return:
            samples: (B, s, s, 2), we sample sxs point for each box, each point has x and y coordinates
            sample_unc: (B, s, s), uncertainties
            sample_conf: (B, s, s, C), confidences
        """
        res = self.voxel_size * self.stride['object']
        grid_size = (
            round((self.lidar_range[3] - self.lidar_range[0]) / res),
            round((self.lidar_range[4] - self.lidar_range[1]) / res),
        )

        # transform box templates to lidar frame
        st = torch.sin(boxes[:, -1]).view(-1, 1, 1)
        ct = torch.cos(boxes[:, -1]).view(-1, 1, 1)
        samples = self.unit_box_template * boxes[:, 4:6].view(-1, 1, 1, 2) * 0.5
        samples = torch.stack([
            samples[..., 0] * ct - samples[..., 1] * st,
            samples[..., 0] * st + samples[..., 1] * ct
        ], dim=-1)
        samples = samples + boxes[:, 1:3].view(-1, 1, 1, 2)  # b sx sy 2

        # convert sample point to map indices
        sx, sy = samples.shape[1:3]
        xy_min = torch.tensor(self.lidar_range[:2], device=samples.device).view(1, 1, 1, 2)
        xy = metric2indices(pad_l(samples - xy_min), res)[..., 1:]
        mask = (xy >= 0).all(dim=-1) & (xy[..., 0] < grid_size[0]) & (xy[..., 1] < grid_size[1])
        xy_masked = xy[mask].T
        batch_indices = torch.tile(boxes[:, 0].view(-1, 1, 1), (1, sx, sy))
        batch_indices = batch_indices[mask].long()

        # retrieve box unc and conf from corresponding maps
        sam_unc = torch.zeros_like(samples[..., 0])
        sam_unc[mask] = unc[batch_indices, xy_masked[0], xy_masked[1]]
        sam_conf = torch.zeros_like(samples)
        sam_conf[mask] = conf[batch_indices, xy_masked[0], xy_masked[1]]

        return samples, sam_unc, sam_conf

    def get_bev_probs(self,
                      boxes: torch.Tensor,
                      ctrs: torch.Tensor,
                      reg: torch.Tensor,
                      res: float = 0.2) -> Tuple:
        # no reg for gaus
        if reg.shape[-1] == 2:
            return None, None
        bs = ctrs[:, 0].max().long().item() + 1
        lr = self.lidar_range
        pts = meshgrid(lr[0], lr[3], lr[1], lr[4], 2, step=res) + res / 2
        pts = torch.tile(pts.unsqueeze(0), (bs, 1, 1, 1))
        pts = pad_l(pad_r(pts))
        for b in range(bs):
            pts[b, :, :, 0] = b
        pts = pts.view(-1,  4).to(boxes.device)
        boxes[:, 3] = 0
        boxes[:, 4:6] *= 1.5
        _, inds = points_in_boxes_gpu(pts, boxes, batch_size=bs)
        pts = pts[inds >= 0, :3]
        evis = draw_sample_prob(ctrs, reg, pts, res,
                                self.distr_r['object'], self.lidar_range, bs,
                                var0=[0.1, 0.1])
        sx = round((self.lidar_range[3] - self.lidar_range[0])/ res)
        sy = round((self.lidar_range[4] - self.lidar_range[1])/ res)
        indices = metric2indices(pts, res).T
        indices[1] = indices[1] - round(self.lidar_range[0] / res)
        indices[2] = indices[2] - round(self.lidar_range[1] / res)
        evidence = torch.zeros((bs, sx, sy, 2), device=ctrs.device)
        evidence[indices[0], indices[1], indices[2]] = evis
        conf, unc = self.evidence_to_conf_unc(evidence)
        return conf, unc
