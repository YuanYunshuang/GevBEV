import functools
import torch, torch_scatter
import MinkowskiEngine as ME
from torch import nn
from model.submodules.utils import minkconv_conv_block, indices2metric, \
    pad_r, linear_last, fuse_batch_indices, meshgrid, metric2indices, \
    weighted_mahalanobis_dists
from ops.utils import points_in_boxes_gpu
from model.losses.common import cross_entroy_with_logits


class BEVBase(nn.Module):
    def __init__(self, cfgs):
        super(BEVBase, self).__init__()
        for name, value in cfgs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)

        self.sampling = getattr(self, 'sampling', None)
        self.sample_pixels = True if self.sampling is not None else False
        self.res = self.stride * self.voxel_size
        self.size = int(self.det_r / self.res)
        r = int(self.det_r / self.voxel_size)
        self.x_max = (r - 1) // self.stride * self.stride        # relevant to ME
        self.x_min = - (self.x_max + self.stride)                # relevant to ME

        feat_dim = self.in_dim
        if hasattr(self, 'conv_kernels'):
            self.get_conv_layer()
            feat_dim = 32

        self.distr_reg = self.get_reg_layer(feat_dim)

        steps = int(self.distr_r / self.res) * 2 + 1
        offset = meshgrid(-self.distr_r, self.distr_r, 2,
                          n_steps=steps).cuda().view(-1, 2)
        self.nbrs = offset[torch.norm(offset, dim=1) < 2].view(1, -1, 2)

        self.centers = None
        self.feat = None
        self.out = {}

    def get_reg_layer(self, in_dim):
        raise NotImplementedError

    def draw_distribution(self, reg):
        raise NotImplementedError

    def loss(self, batch_dict):
        raise NotImplementedError

    def forward(self, batch_dict):
        stensor3d = batch_dict['compression'][f'p{self.stride}']
        coor = fuse_batch_indices(stensor3d.C, batch_dict['num_cav'])
        obs_mask = self.get_obs_mask(coor)
        coor, feat = self.get_distr_samples(coor[:, :3], stensor3d.F, batch_dict)

        if hasattr(self, 'convs'):
            stensor2d = ME.SparseTensor(
                coordinates=coor.contiguous(),
                features=feat,
                tensor_stride=[self.stride] * 2
            )
            stensor2d = self.convs(stensor2d)
            # after coordinate expansion, some coordinates will exceed the maximum detection
            # range, therefore they are removed here.
            mask = torch.logical_and(
                stensor2d.C[:, 1:] >= self.x_min,
                stensor2d.C[:, 1:] <= self.x_max,
            ).all(dim=-1)
            # mask = (stensor2d.C[:, 1:].abs() < (self.det_r / self.voxel_size)).all(dim=-1)
            coor = stensor2d.C[mask]
            feat = stensor2d.F[mask]

        # todo: barely fuse voxels with batch indices might cause overlapped voxels even it has
        #  a low chance, we temporarily average them.
        coor, indices = coor.unique(dim=0, return_inverse=True)
        feat = torch_scatter.scatter_mean(feat, indices, dim=0)
        self.centers = indices2metric(coor, self.voxel_size)
        self.feat = feat
        reg = self.distr_reg(feat)

        self.out = {
            'reg': reg,
        }

        evidence = self.draw_distribution(reg)

        batch_dict[f'distr_{self.name}'] = {
            'evidence': evidence,
            'obs_mask': obs_mask,
            'centers': self.centers,
            'reg': self.out['reg']
        }

    def get_obs_mask(self, coor):
        voxel_new = coor[:, 1:3].view(-1, 1, 2) + self.nbrs
        size = self.size
        xy = (torch.floor(voxel_new / self.res) + size).view(-1, 2)
        mask = torch.logical_and(xy >= 0, xy < (size * 2)).all(dim=1)
        xy = xy[mask].long().T
        batch_indices = torch.tile(coor[:, 0].view(-1, 1), (1, self.nbrs.shape[1])).view(-1)
        batch_indices = batch_indices[mask].long()
        batch_size = coor[:, 0].max().int() + 1
        obs_mask = torch.zeros((batch_size, size * 2, size * 2),
                               device=coor.device)
        obs_mask[batch_indices, xy[0], xy[1]] = 1
        return obs_mask.bool()

    def get_distr_samples(self, coor_in, feat_in, batch_dict):
        if self.sample_pixels:
            sampling_fn = getattr(self, f'sample_with_{self.sampling}')
            coor_out, feat_out = sampling_fn(coor_in, feat_in, batch_dict)
        else:
            coor_out = coor_in[:, :3]
            feat_out = feat_in

        if self.training:
            keep = torch.rand_like(feat_out[:, 0]) > 0.5
            coor_out = coor_out[keep]
            feat_out = feat_out[keep]

        return coor_out, feat_out

    def sample_with_boxes(self, coor_in, feat_in, batch_dict):
        coor_metric = indices2metric(coor_in, self.voxel_size)
        coor_metric = pad_r(coor_metric)
        boxes_ = batch_dict['detection']['pred_boxes'].clone()
        boxes_[:, 3] = 0
        _, box_idx_of_pts = points_in_boxes_gpu(
            coor_metric, boxes_, batch_dict['batch_size']
        )
        in_mask = box_idx_of_pts >= 0
        coor_out = coor_in[in_mask][:, :3]
        feat = feat_in[in_mask]
        return coor_out, feat

    def sample_with_road(self, coor_in, feat_in, batch_dict):
        evidence = batch_dict['distr_surface']['evidence']
        road_mask = torch.argmax(evidence, dim=-1).bool()
        sm = road_mask.shape[1]
        ratio_diff = self.size * 2 / sm
        coor = coor_in.clone().long().T
        coor[1:] = torch.floor((coor[1:] + self.size) / ratio_diff).long()
        mask1 = torch.logical_and(coor[1:] >= 0, coor[1:] < sm).all(dim=0)
        mask2 = road_mask[coor[0, mask1], coor[1, mask1], coor[2, mask1]]
        coor_out = coor_in[mask1][mask2]
        feat_out = feat_in[mask1][mask2]
        return coor_out, feat_out

    @torch.no_grad()
    def get_bev_pts_with_boxes(self, batch_dict):
        # get bev points
        bev_pts = pad_r(self.centers.clone())
        bev_pts = torch.tile(bev_pts.unsqueeze(1), (1, 3, 1))
        bev_pts[:, :, 1:3] = bev_pts[:, :, 1:3] + \
                          torch.randn_like(bev_pts[:, :, 1:3])
        bev_pts[:, :, 1:3] = (bev_pts[:, :, 1:3] / self.res).int()
        bev_pts = torch.unique(bev_pts.view(-1, 4), dim=0)
        bev_pts[:, 1:3] = bev_pts[:, 1:3] * self.res

        pred_boxes = batch_dict['detection']['pred_boxes']
        pred_boxes_ext = pred_boxes.clone()
        pred_boxes_ext[:, 4:7] = pred_boxes_ext[:, 4:7] + 4
        pred_boxes_ext[:, 3] = 0
        _, box_idx_of_pts = points_in_boxes_gpu(
            bev_pts, pred_boxes_ext, batch_size=batch_dict['batch_size']
        )
        bev_pts = bev_pts[box_idx_of_pts >= 0]
        # down sample
        bev_pts = bev_pts[torch.rand(len(bev_pts)) > 0.5]

        return bev_pts[:, :3]

    @torch.no_grad()
    def bev_pts_to_indices(self, bev_pts):
        ixy = metric2indices(bev_pts[:, :3], self.res).long()
        ixy[:, 1:] += self.size
        mask = torch.logical_and(ixy[:, 1:] >= 0, ixy[:, 1:] < self.size * 2).all(dim=-1)
        indices = ixy[mask]
        return indices, mask

    @torch.no_grad()
    def get_tgt(self, batch_dict):
        if self.sample_pixels:
            bev_pts = self.get_bev_pts_with_boxes(batch_dict)
            tgt = self.get_tgt_with_boxes(
                bev_pts,
                batch_dict['target_boxes'],
                batch_dict['batch_size']
            )
            indices, mask = self.bev_pts_to_indices(bev_pts)
            tgt = tgt[mask]
            bev_pts = bev_pts[mask]
        else:
            bev_pts = fuse_batch_indices(batch_dict['target_bev_pts'],
                                         batch_dict['num_cav'])
            indices, mask = self.bev_pts_to_indices(bev_pts)
            tgt = bev_pts[mask, 3]
            bev_pts = bev_pts[mask]
        return tgt, indices.T, bev_pts

    @torch.no_grad()
    def get_tgt_with_boxes(self, bev_pts, target_boxes, bs):
        boxes = target_boxes.clone()
        boxes[:, 3] = 0
        pts = pad_r(bev_pts)
        _, box_idx_of_pts = points_in_boxes_gpu(
            pts, boxes, batch_size=bs
        )
        pos = box_idx_of_pts >= 0

        return pos