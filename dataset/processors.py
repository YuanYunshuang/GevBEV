import logging, os
import torch, torch_scatter
import torch.nn.functional as F
import numpy as np
from utils import pclib, box_utils, vislib
from torch.distributions.multivariate_normal import _batch_mahalanobis
from model.losses import edl
from model.submodules.utils import meshgrid, metric2indices, draw_sample_prob, pad_l
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
                 det_r,
                 distr_r,
                 visualization=False,
                 vis_dir=None,
                 edl=True):
        super(DistributionPostProcess, self).__init__()
        self.det_r = det_r
        self.distr_r = distr_r
        self.voxel_size = voxel_size[0] if isinstance(voxel_size, list) else voxel_size
        self.stride = stride
        self.vis = visualization
        self.vis_dir = vis_dir
        self.edl = edl

        self.n_sam_per_dim = 29
        self.uni_box_samples = meshgrid(
            -1.4, 1.4, dim=2, n_steps=self.n_sam_per_dim
        ).view(1, self.n_sam_per_dim, self.n_sam_per_dim, 2).cuda()

        self.out = {}

    def __call__(self, batch_dict):
        self.surface(batch_dict)
        self.object(batch_dict)
        if self.vis:
            self.visualization()

    def surface(self, batch_dict):
        res = self.voxel_size * self.stride['surface']
        grid_size = int(self.det_r / res * 2)
        evidence = batch_dict['distr_surface']['evidence']
        obs_mask = batch_dict['distr_surface']['obs_mask']
        
        conf, unc = self.evidence_to_conf_unc(evidence)
        
        s = batch_dict['bevmap_static'].shape[2] // grid_size
        road_bev = batch_dict['bevmap_static'][:, ::s, ::s].bool()
        
        self.out = {
            'frame_id': batch_dict['frame_id'],
            'road_confidence': conf,
            'road_bev': road_bev,
            'road_uncertainty': unc.squeeze(),
            'road_obs_mask': obs_mask
        }

        if self.vis:
            points = batch_dict['xyz']
            num_cav = batch_dict['num_cav'][0]
            points_idx0 = batch_dict['in_data'].C[:, 0] < num_cav
            points = points[points_idx0]
            self.out['points'] = points

    def object(self, batch_dict):
        pred_boxes = batch_dict['detection']['pred_boxes']
        gt_boxes = batch_dict['target_boxes']
        evidence = batch_dict['distr_object']['evidence']
        obs_mask = batch_dict['distr_object']['obs_mask']

        conf, unc = self.evidence_to_conf_unc(evidence)

        # centers = batch_dict['distr_object']['centers']
        # reg = batch_dict['distr_object']['reg'].relu()
        # pred_sam_coor, pred_box_unc, pred_box_conf = \
        #     self.get_sample_probs_metirc(pred_boxes, centers, reg)
        # gt_sam_coor, gt_box_unc, gt_box_conf = \
        #     self.get_sample_probs_metirc(gt_boxes, centers, reg)

        pred_sam_coor, pred_box_unc, pred_box_conf = \
            self.get_sample_probs_pixel(pred_boxes, unc, conf)
        gt_sam_coor, gt_box_unc, gt_box_conf = \
            self.get_sample_probs_pixel(gt_boxes, unc, conf)

        self.out.update({
            'box_bev_unc': unc,
            'box_bev_conf': conf,
            'box_obs_mask': obs_mask,
            'pred_boxes': pred_boxes,
            'gt_boxes': gt_boxes,
            'pred_box_samples': pred_sam_coor,
            'pred_box_unc': pred_box_unc,
            'pred_box_conf': pred_box_conf,
            'gt_box_samples': gt_sam_coor,
            'gt_box_unc': gt_box_unc,
            'gt_box_conf': gt_box_conf
        })
    
    def visualization(self):
        out_dict = self.out
        fn = '_'.join(out_dict['frame_id'][0])
        # surface
        res = self.voxel_size * self.stride['surface']
        grid_size = int(self.det_r / res * 2)
        points = out_dict.pop('points').cpu().numpy()
        road_bev = out_dict['road_bev'][0].cpu().numpy()
        confs_np = out_dict['road_confidence'][0].cpu().numpy()
        obs_msk = out_dict['road_obs_mask'][0].cpu().numpy()
        unc = out_dict['road_uncertainty'][0].cpu().numpy()

        road_mask = np.argmax(confs_np, axis=-1).astype(bool)
        road_points = np.stack(np.where(road_mask), axis=-1)
        road_points = (road_points * res) - self.det_r + res / 2
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
        gt_bevmap[..., 1] = road_bev

        fig = plt.figure(figsize=(13, 6))
        axs = fig.subplots(1, 3)
        axs[0].imshow(1 - unc.T[::-1], cmap='hot')
        axs[1].imshow(bevmap.astype(np.uint8).transpose(1, 0, 2)[::-1])
        axs[2].imshow(gt_bevmap.astype(np.uint8).transpose(1, 0, 2)[::-1])
        fig.suptitle(f"road: {iou_road:.2f}", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, fn) + '_2.png')
        plt.close()

        # box
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

        fig = plt.figure(figsize=(8, 8))
        ax = fig.subplots()
        ax.set_facecolor((0.0, 0.0, 0.0))
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

        ax.set_xlim([-50, 50])
        ax.set_ylim([-50, 50])
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, fn) + '_1.png')
        plt.close()

        # box bev
        box_bev_conf = self.out['box_bev_conf'][0, :, :, 1].cpu().numpy().T
        plt.imshow(box_bev_conf[::-1], cmap='hot', vmin=0, vmax=1)
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
            conf = torch.sqrt(conf * (1 - unc))
            unc = unc.squeeze(dim=-1)
        else:
            # use entropy as uncertainty
            entropy = -evidence * torch.log2(evidence)
            unc = entropy.sum(dim=-1)
            conf = torch.sqrt(evidence * (1 - unc.unsqueeze(-1)))
        return conf, unc

    def get_sample_probs_metirc(self, boxes, ctrs, ctrs_reg):
        """resample points from box parameters and draw probs from box prob map"""
        res = self.voxel_size * self.stride['object']
        grid_size = int(self.det_r / res * 2)
        # transform to lidar frame
        st = torch.sin(boxes[:, -1]).view(-1, 1, 1)
        ct = torch.cos(boxes[:, -1]).view(-1, 1, 1)
        samples = self.uni_box_samples * boxes[:, 4:6].view(-1, 1, 1, 2) * 0.5
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
        xy = samidx[..., 1:] + grid_size / 2
        mask = torch.logical_and(xy >= 0, xy < grid_size * 2).all(dim=-1)

        evis = draw_sample_prob(ctrs, ctrs_reg, samples[mask],
                                 res, self.distr_r['object'], self.det_r,
                                 ctrs[:, 0].max().long().item() + 1,
                                 var0=[0.1, 0.1])
        sam_evis = torch.zeros_like(samples[..., :2])
        sam_evis[mask] = evis
        sam_conf, sam_unc = self.evidence_to_conf_unc(sam_evis)
        return samples[..., 1:], sam_unc, sam_conf

    def get_sample_probs_pixel(self, boxes, unc, conf):
        res = self.voxel_size * self.stride['object']
        grid_size = int(self.det_r / res * 2)
        # transform to lidar frame
        st = torch.sin(boxes[:, -1]).view(-1, 1, 1)
        ct = torch.cos(boxes[:, -1]).view(-1, 1, 1)
        samples = self.uni_box_samples
        samples = samples * boxes[:, 4:6].view(-1, 1, 1, 2) * 0.5
        samples = torch.stack([
            samples[..., 0] * ct - samples[..., 1] * st,
            samples[..., 0] * st + samples[..., 1] * ct
        ], dim=-1)
        samples = samples + boxes[:, 1:3].view(-1, 1, 1, 2)  # b s s 2
        s = samples.shape[1]
        xy = metric2indices(pad_l(samples + self.det_r), res)[..., 1:]
        mask = torch.logical_and(xy >= 0, xy < grid_size).all(dim=-1)
        xy_masked = xy[mask].T
        batch_indices = torch.tile(boxes[:, 0].view(-1, 1, 1), (1, s, s))
        batch_indices = batch_indices[mask].long()
        sam_unc = torch.zeros_like(samples[..., 0])
        sam_unc[mask] = unc[batch_indices, xy_masked[0], xy_masked[1]]
        sam_conf = torch.zeros_like(samples)
        sam_conf[mask] = conf[batch_indices, xy_masked[0], xy_masked[1]]
        return samples, sam_unc, sam_conf