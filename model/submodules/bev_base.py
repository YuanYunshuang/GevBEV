import functools
import torch_scatter
from model.submodules.utils import *
from ops.utils import points_in_boxes_gpu
from model.losses.common import cross_entroy_with_logits


class HBEVBase(nn.Module):
    DISTR_CLS = None
    def __init__(self, cfgs):
        super(HBEVBase, self).__init__()
        for k, v in cfgs['args'].items():
            setattr(self, k, v)
        self.heads = []
        for cfg in cfgs['heads']:
            k = list(cfg.keys())[0]
            cfg[k].update(cfgs['args'])
            setattr(self, k, self.DISTR_CLS(cfg[k]))
            self.heads.append(k)

        self.convs = []
        for k, conv_args in cfgs['convs'].items():
            self.convs.append(k)
            setattr(self, f'convs_{k}', self.get_conv_layer(conv_args))
            stride = int(k[1])

            if getattr(self, 'det_r', False):
                lr = [-self.det_r, -self.det_r, 0, self.det_r, self.det_r, 0]
            elif getattr(self, 'lidar_range', False):
                lr = self.lidar_range
            else:
                raise NotImplementedError
            setattr(self, f'mink_xylim_{k}', mink_coor_limit(lr, self.voxel_size, stride))  # relevant to ME

    def get_conv_layer(self, args):
        minkconv_layer = functools.partial(
            minkconv_conv_block, d=2, bn_momentum=0.1,
        )
        layers = [minkconv_layer(args['in_dim'], 32, args['kernels'][0], 1,
                                 expand_coordinates=args['expand_coordinates'])]
        for ks in args['kernels'][1:]:
            layers.append(minkconv_layer(32, 32, ks, 1,
                                         expand_coordinates=args['expand_coordinates']))
        return nn.Sequential(*layers)

    def forward(self, batch_dict):
        batch_dict['distr_conv_out'] = {}
        for k in self.convs:
            stride = int(k[1])
            stensor3d = batch_dict['compression'][k]
            coor = stensor3d.C[:, :3]
            feat = stensor3d.F
            if getattr(self, "cpm_option", "none") == "none":
                # fuse at feature level
                coor, feat = self.fuse_coor_feat(coor, feat, batch_dict['num_cav'])
            obs_mask = self.get_obs_mask(fuse_batch_indices(stensor3d.C, batch_dict['num_cav']), stride)

            stensor2d = ME.SparseTensor(
                coordinates=coor.contiguous(),
                features=feat,
                tensor_stride=[stride] * 2
            )
            stensor2d = getattr(self, f'convs_{k}')(stensor2d)
            # after coordinate expansion, some coordinates will exceed the maximum detection
            # range, therefore they are removed here.
            xylim = getattr(self, f'mink_xylim_{k}')
            mask = (stensor2d.C[:, 1] > xylim[0]) & (stensor2d.C[:, 1] <= xylim[1]) & \
                   (stensor2d.C[:, 2] > xylim[2]) & (stensor2d.C[:, 2] <= xylim[3])

            # mask = (stensor2d.C[:, 1:].abs() < (self.det_r / self.voxel_size)).all(dim=-1)
            coor = stensor2d.C[mask]
            feat = stensor2d.F[mask]
            if getattr(self, "cpm_option", "none") != "none" and self.training:
                # fuse at the evidence level
                coor, feat = self.fuse_coor_feat(coor, feat, batch_dict['num_cav'])
            batch_dict['distr_conv_out'][k] = {
                'coor': coor,
                'feat': feat,
                'obs_mask': obs_mask
            }

        for h in self.heads:
            getattr(self, h)(batch_dict)

    def fuse_coor_feat(self, coor, feat, num_cavs):
        coor = fuse_batch_indices(coor, num_cavs)
        # todo: barely fuse voxels with batch indices might cause overlapped voxels even it has
        #  a low chance, we temporarily average them.
        coor, indices = coor[:, :3].unique(dim=0, return_inverse=True)
        feat = torch_scatter.scatter_mean(feat, indices, dim=0)
        return coor, feat

    def get_obs_mask(self, coor, stride):
        res = stride * self.voxel_size
        if getattr(self, 'det_r', False):
            sizex = round(self.det_r * 2 / res)
            sizey = sizex
        else:
            sizex = round((self.lidar_range[3] - self.lidar_range[0]) / res)
            sizey = round((self.lidar_range[4] - self.lidar_range[1]) / res)
        steps = round(self.distr_r / res) * 2 + 1
        offset = meshgrid(-self.distr_r, self.distr_r, 2,
                          n_steps=steps).to(coor.device).view(-1, 2)
        nbrs = offset[torch.norm(offset, dim=1) < 2].view(1, -1, 2)
        voxel_new = coor[:, 1:3].view(-1, 1, 2) / stride + nbrs
        x = (torch.floor(voxel_new [..., 0] / res) - round(self.lidar_range[0] / res)).view(-1)
        y = (torch.floor(voxel_new [..., 1] / res) - round(self.lidar_range[1] / res)).view(-1)
        maskx = torch.logical_and(x >= 0, x < sizex)
        masky = torch.logical_and(y >= 0, y < sizey)
        mask = torch.logical_and(maskx, masky)
        x = x[mask].long()
        y = y[mask].long()
        batch_indices = torch.tile(coor[:, 0].view(-1, 1), (1, nbrs.shape[1])).view(-1)
        batch_indices = batch_indices[mask].long()
        batch_size = coor[:, 0].max().int() + 1
        obs_mask = torch.zeros((batch_size, sizex, sizey),
                               device=coor.device)
        obs_mask[batch_indices, x, y] = 1
        return obs_mask.bool()

    def loss(self, batch_dict):
        loss = 0
        loss_dict = {}
        for h in self.heads:
            l, ldict = getattr(self, h).loss(batch_dict)
            loss = loss + l
            loss_dict.update(ldict)
        return loss, loss_dict


class BEVBase(nn.Module):
    def __init__(self, cfgs):
        super(BEVBase, self).__init__()
        for name, value in cfgs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)

        self.sampling = getattr(self, 'sampling', None)
        self.sample_pixels = True if self.sampling is not None else False
        self.res = self.stride * self.voxel_size
        if getattr(self, 'det_r', False):
            lr = [-self.det_r, -self.det_r, 0, self.det_r, self.det_r, 0]
        elif getattr(self, 'lidar_range', False):
            lr = self.lidar_range
        else:
            raise NotImplementedError
        self.mink_xylim = mink_coor_limit(lr, self.voxel_size, self.stride)
        self.size_x = round((lr[3] - lr[0]) / self.res)
        self.size_y = round((lr[4] - lr[1]) / self.res)
        self.offset_sz_x = round(lr[0] / self.res)
        self.offset_sz_y = round(lr[1] / self.res)

        feat_dim = 32
        self.distr_reg = self.get_reg_layer(feat_dim)

        self.centers = None
        self.feat = None
        self.out = {}

    def get_reg_layer(self, in_dim):
        raise NotImplementedError

    def draw_distribution(self, batch_dict):
        raise NotImplementedError

    def loss(self, batch_dict):
        raise NotImplementedError

    def forward(self, batch_dict):
        conv_out = batch_dict['distr_conv_out'][f'p{self.stride}']
        coor = conv_out['coor']
        feat = conv_out['feat']
        if self.training:
            coor, feat = self.down_sample(coor, feat)

        self.centers = indices2metric(coor, self.voxel_size)
        self.feat = feat
        reg = self.distr_reg(feat)

        self.out = {
            'reg': reg,
        }

        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(10, 10))
        # mask = self.centers[:, 0] == 0
        # ctr = self.centers[mask, 1:].detach().cpu().numpy()
        #
        # alpha = reg[mask, :2].relu() + 1
        # S = torch.sum(alpha, dim=-1, keepdim=True)
        # conf = torch.div(alpha, S)
        # colors = conf[:, 1].detach().cpu().numpy()
        # plt.scatter(ctr[:, 0], ctr[:, 1], cmap='jet', c=colors, edgecolors=None, marker='.', s=2, vmin=0, vmax=1)
        # plt.show()
        # plt.close()

        if not self.training:
            evidence = self.draw_distribution(batch_dict)

            # visualization
            # alpha = evidence + 1
            # S = torch.sum(alpha, dim=-1, keepdim=True)
            # conf = torch.div(alpha, S)
            # plt.imshow(conf[0, :, :, 1].detach().cpu().numpy())
            # plt.show()
            # plt.close()

            batch_dict[f'distr_{self.name}'] = {
                'evidence': evidence,
                'obs_mask': conv_out['obs_mask'],
                'Nall': self.out.get('Nall', None),
                'Nsel': self.out.get('Nsel', None)
            }

    def down_sample(self, coor, feat):
        keep = torch.rand_like(feat[:, 0]) > 0.5
        coor = coor[keep]
        feat = feat[keep]

        return coor, feat

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
        sm = road_mask.shape[1:]
        ratio_diff_x, ratio_diff_y = self.size_x / sm[0], self.size_y / sm[1]
        coor = coor_in.clone().long().T
        coor[1] = torch.floor((coor[1] - self.offset_sz_x) / ratio_diff_x).long()
        coor[2] = torch.floor((coor[2] - self.offset_sz_y) / ratio_diff_y).long()
        mask1x = torch.logical_and(coor[1] >= 0, coor[1] < sm[0])
        mask1y = torch.logical_and(coor[2] >= 0, coor[2] < sm[1])
        mask1 = torch.logical_and(mask1x, mask1y)
        mask2 = road_mask[coor[0, mask1], coor[1, mask1], coor[2, mask1]]
        coor_out = coor_in[mask1][mask2]
        feat_out = feat_in[mask1][mask2]
        return coor_out, feat_out

    @torch.no_grad()
    def get_bev_pts_with_boxes(self, batch_dict):
        # get bev-opv2v points
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
    def pts_to_masked_indices(self, pts):
        ixy = metric2indices(pts[:, :3], self.res).long()
        ixy[:, 1] -= self.offset_sz_x
        ixy[:, 2] -= self.offset_sz_y
        maskx = torch.logical_and(ixy[:, 1] >= 0, ixy[:, 1] < self.size_x)
        masky = torch.logical_and(ixy[:, 2] >= 0, ixy[:, 2] < self.size_y)
        mask = torch.logical_and(maskx, masky)
        indices = ixy[mask]
        return indices, mask

    @torch.no_grad()
    def sample_tgt_pts(self, obs_mask, discrete=False):
        tgt_pts = self.centers.clone()
        if not discrete:
            tgt_pts[:, 1:3] = tgt_pts[:, 1:3] + torch.randn_like(tgt_pts[:, 1:3]) * 3
        indices, mask = self.pts_to_masked_indices(tgt_pts)
        tgt_pts = tgt_pts[mask]
        mask = obs_mask[indices[:, 0], indices[:, 1], indices[:, 2]]
        return tgt_pts[mask], indices[mask]

    @torch.no_grad()
    def downsample_tgt_pts(self, tgt_label, max_sam):
        selected = torch.ones_like(tgt_label.bool())
        pos = tgt_label == 1
        if pos.sum() > max_sam:
            mask = torch.rand_like(tgt_label[pos].float()) < max_sam / pos.sum()
            selected[pos] = mask

        neg = tgt_label == 0
        if neg.sum() > max_sam:
            mask = torch.rand_like(tgt_label[neg].float()) < max_sam / neg.sum()
            selected[neg] = mask
        return selected

    @torch.no_grad()
    def get_tgt(self, batch_dict, discrete=False):
        obs_mask = batch_dict['distr_conv_out'][f'p{self.stride}']['obs_mask']

        if self.name == 'surface':
            tgt_bev_pts = fuse_batch_indices(batch_dict['target_bev_pts'],
                                             batch_dict['num_cav'])
            tgt_pts = tgt_bev_pts[:, :3]
            indices, mask = self.pts_to_masked_indices(tgt_pts)
            tgt_pts = tgt_pts[mask]
            tgt_label = tgt_bev_pts[mask, 3]
        else:  # object
            tgt_pts, indices = self.sample_tgt_pts(obs_mask, discrete)
            boxes = batch_dict['target_boxes'].clone()
            boxes[:, 3] = 0
            pts = pad_r(tgt_pts)
            _, box_idx_of_pts = points_in_boxes_gpu(
                pts, boxes, batch_size=batch_dict['batch_size']
            )
            boxes[:, 4:6] *= 4
            _, box_idx_of_pts2 = points_in_boxes_gpu(
                pts, boxes, batch_size=batch_dict['batch_size']
            )
            tgt_label = - (box_idx_of_pts2 >=0).int()
            tgt_label[box_idx_of_pts >= 0] = 1

        n_sam = 3000 if self.name=='surface' else len(batch_dict['target_boxes']) * 50
        mask = self.downsample_tgt_pts(tgt_label, max_sam=n_sam)
        tgt_label = tgt_label > 0
        return tgt_pts[mask], tgt_label[mask], indices[mask].T


