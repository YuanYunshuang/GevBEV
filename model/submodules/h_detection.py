import torch_scatter
from model.submodules.utils import *
from ops.iou3d_nms_utils import nms_gpu, boxes_iou_bev, aligned_boxes_iou3d_gpu, \
    boxes_iou3d_gpu
from ops.utils import points_in_boxes_gpu
from model.losses.common import weighted_smooth_l1_loss, \
    sigmoid_binary_cross_entropy

pi = 3.1415926


class HDetection(nn.Module):
    def __init__(self, cfgs):
        super(HDetection, self).__init__()
        for cfg in cfgs:
            name = list(cfg.keys())[0]
            value = cfg[name]
            if name not in ["model", "__class__"]:
                setattr(self, name, value)

        self.det_s1 = DetectionS1(self.s1)
        self.det_s2 = None
        if 's2' in cfgs:
            self.det_s2 = DetectionS2(self.s2)

    def forward(self, batch_dict):
        batch_dict['detection'] = {}
        boxes = self.det_s1(batch_dict)
        if self.det_s2 is not None:
            boxes = self.det_s2(batch_dict)
        batch_dict['detection']['pred_boxes'] = boxes

    def loss(self, batch_dict):
        loss_dict = {}
        loss, loss_dict1 = self.det_s1.loss(batch_dict)
        loss_dict.update(loss_dict1)
        if self.det_s2 is not None:
            loss2, loss_dict2 = self.det_s2.loss(batch_dict)
            loss = loss + loss2
            loss_dict.update(loss_dict2)
        return loss, loss_dict


class DetectionS1(nn.Module):
    def __init__(self, cfgs):
        super(DetectionS1, self).__init__()
        for k, v in cfgs.items():
            setattr(self, k, v)
        self.device = getattr(self, 'device', 'cuda')
        if getattr(self, 'det_r', False):
            self.grid_size = int(self.det_r / self.voxel_size / self.stride * 2)
            self.rx = int(self.det_r / self.voxel_size)
            self.x_max = (self.rx - 1) // self.stride * self.stride  # relevant to ME
            self.x_min = - (self.x_max + self.stride)  # relevant to ME
            self.ry = self.rx
            self.y_max = self.x_max
            self.y_min = self.x_min
        elif getattr(self, 'lidar_range', False):
            lr = self.lidar_range
            self.grid_size = (
                int((lr[3] - lr[0]) / self.voxel_size / self.stride),
                int((lr[4] - lr[1]) / self.voxel_size / self.stride),
            )
            self.rx = int(lr[3] - lr[0] / self.voxel_size)
            self.x_max = (self.rx - 1) // self.stride * self.stride  # relevant to ME
            self.x_min = - (self.x_max + self.stride)  # relevant to ME
            self.ry = int(lr[4] - lr[1] / self.voxel_size)
            self.y_max = (self.ry - 1) // self.stride * self.stride  # relevant to ME
            self.y_min = - (self.y_max + self.stride)  # relevant to ME
        else:
            raise NotImplementedError

        self.anchors = self.generate_anchors().to(self.device)
        # intermediate result
        self.xy = None
        self.coor = None
        self.out = {}

        ks = int(0.8 / self.voxel_size / self.stride) * 2 + 1
        self.convs = nn.Sequential(
            minkconv_conv_block(self.in_dim, 64, ks, 1, d=2, bn_momentum=0.1,
                                expand_coordinates=True),
            minkconv_conv_block(64, 64, ks, 1, d=2, bn_momentum=0.1,
                                expand_coordinates=True)
        )
        self.cls = linear_last(64, 32, 2)
        self.scores = linear_last(64, 32, 3 * 2)         # iou, dir1, dir2
        self.reg = linear_last(64, 32, 10 * 2)           # xyzlwh, ct1, st1, ct2, st2

    def generate_anchors(self):
        xy = meshgrid(self.x_min, self.rx, self.y_min, self.ry, 2, step=self.stride,
                      ) * self.voxel_size  # h w 2
        h, w, _ = xy.shape
        anchors = torch.zeros((h, w, len(self.box_angles), 7))  # h w 2 7
        anchors[..., :2] = torch.tile(xy.unsqueeze(2), (1, 1, 2, 1))  # set x y
        anchors[..., 2] = self.box_z  # set z
        anchors[..., 3] = self.box_dim[0]  # set l
        anchors[..., 4] = self.box_dim[1]  # set w
        anchors[..., 5] = self.box_dim[2]  # set h
        for i, a in enumerate(self.box_angles):
            anchors[:, :, i, 6] = a / 180 * pi  # set angle

        return anchors

    def forward(self, batch_dict):
        self.points = batch_dict['xyz'][
                batch_dict['in_data'].C[:, 0] == 0].cpu().numpy()
        self.gt_boxes = batch_dict['target_boxes'][
            batch_dict['target_boxes'][:, 0] == 0,
        1:].cpu().numpy()
        stensor3d = batch_dict['compression'].get(f'p{self.stride}')
        stensor2d = ME.SparseTensor(
            coordinates=stensor3d.C[:, :3].contiguous(),
            features=stensor3d.F,
            tensor_stride=[self.stride] * 2
        )
        self.update_coords(stensor2d.C)
        feat = stensor2d.F

        self.out['cls'] = self.cls(feat)
        self.out['scores'] = self.scores(feat)
        self.out['reg'] = self.reg(feat)

        self.get_stage_one_boxes()
        boxes_fused, scores_fused = self.box_fusion(batch_dict['num_cav'])
        batch_dict['detection']['boxes_fused'] = boxes_fused
        return boxes_fused

    def update_coords(self, coor):
        xy = coor.float().T
        xy[1] = (xy[1] + self.x_min) / self.stride
        xy[2] = (xy[2] + self.y_min) / self.stride
        self.xy = xy.long()
        coor_ = coor.float()
        coor_[:, 1:] = coor_[:, 1:] * self.voxel_size
        self.coor = coor_.T

    def get_stage_one_boxes(self):
        dec_boxes = self.decode_boxes()
        ious_pred = self.out['scores'][:, :2].sigmoid()
        cls_scores = self.out['cls'].sigmoid()
        boxes = []
        ious = []
        scrs = []
        for b in self.xy[0].unique():
            mask = self.xy[0] == b
            cur_boxes = dec_boxes[mask].view(-1, 7)
            cur_scores = cls_scores[mask].view(-1)
            cur_ious = ious_pred[mask].view(-1)
            # remove abnormal boxes
            mask = torch.logical_and(
                cur_boxes[:, 3:6] > 1,
                cur_boxes[:, 3:6] < 10
            ).all(dim=-1)

            keep = torch.logical_and(cur_scores > 0.5, mask)
            if keep.sum() == 0:
                boxes.append(torch.empty((0, 7), device=cur_boxes.device))
                scrs.append(torch.empty((0,), device=cur_boxes.device))
                ious.append(torch.empty((0,), device=cur_boxes.device))
                continue
            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_ious = cur_ious[keep]

            cur_scores_rectified = cur_scores * cur_ious ** 4
            keep = nms_gpu(cur_boxes, cur_scores_rectified,
                           thresh=0.01, pre_maxsize=500)
            boxes.append(cur_boxes[keep])
            scrs.append(cur_scores[keep])
            ious.append(cur_ious[keep])

        self.out['pred_box'] = boxes
        self.out['pred_scr'] = scrs
        self.out['pred_iou'] = ious

    def box_fusion(self, num_cavs):
        boxes = self.out['pred_box']
        scores = self.out['pred_scr']
        ious = self.out['pred_iou']
        boxes_fused = []
        scores_fused = []
        idx_start = 0
        for b, num in enumerate(num_cavs):
            idx_end = idx_start + num
            cur_boxes = torch.cat(boxes[idx_start:idx_end], dim=0)
            cur_scores = torch.cat(scores[idx_start:idx_end], dim=0)
            cur_ious = torch.cat(ious[idx_start:idx_end], dim=0)
            cur_scores_rectified = cur_scores * cur_ious ** 4
            idx_start = idx_end
            if len(cur_boxes) == 0:
                boxes_fused.append(torch.zeros((0, 8), device=cur_boxes.device))
                scores_fused.append(torch.zeros((0,), device=cur_boxes.device))
                continue
            keep = nms_gpu(cur_boxes, cur_scores_rectified,
                           thresh=0.01, pre_maxsize=100)
            bf = cur_boxes[keep]
            sf = cur_scores[keep]
            boxes_fused.append(torch.cat(
                [torch.ones_like(sf).view(-1, 1) * b, bf], dim=-1)
            )
            scores_fused.append(sf)
        boxes_fused = torch.cat(boxes_fused, dim=0)
        scores_fused = torch.cat(scores_fused, dim=0)
        return boxes_fused, scores_fused

    def loss(self, batch_dict):
        cls_tgt, iou_tgt, dir_tgt, reg_tgt = self.get_target(batch_dict)
        cared = cls_tgt >= 0
        # cls loss
        loss_cls = sigmoid_binary_cross_entropy(self.out['cls'][cared],
                                                cls_tgt[cared]).mean()
        loss_dict = {'bx_cls': loss_cls}
        loss = loss_cls
        pos = cls_tgt == 1
        if pos.sum() > 0:
            # reg loss
            reg_src = self.out['reg'].view(-1, 2, 10)[pos]
            loss_reg = weighted_smooth_l1_loss(reg_src, reg_tgt).mean()
            # score loss
            ious = self.out['scores'][:, :2][cared].sigmoid()
            dirs = self.out['scores'][:, 2:].sigmoid().view(-1, 2, 2)[pos]
            loss_iou = weighted_smooth_l1_loss(ious, iou_tgt[cared]).mean()
            loss_dir = weighted_smooth_l1_loss(dirs, dir_tgt).mean()

            loss = loss + loss_iou + loss_dir + loss_reg
            loss_dict.update({
                'bx_iou': loss_iou,
                'bx_dir': loss_dir,
                'bx_reg': loss_reg,
            })
        return loss, loss_dict

    @torch.no_grad()
    def get_target(self, batch_dict):
        gt_boxes = batch_dict['target_boxes']

        # remove boxes with observation points < 3
        raw_pts = self.get_raw_points(batch_dict)
        boxes_decomposed, box_idxs_of_pts = points_in_boxes_gpu(
            raw_pts, gt_boxes, batch_dict['batch_size']
        )
        box_idx = [i for i in box_idxs_of_pts[box_idxs_of_pts >= 0].unique() if
                      (box_idxs_of_pts == i).sum() > 3]
        gt_boxes = boxes_decomposed[box_idx, :]

        # get batch indices for boxes
        batch_size = sum(batch_dict['num_cav'])
        box_indices = []
        for i, n in enumerate(batch_dict['num_cav']):
            box_indices.extend([i] * n)

        batch_anchors = self.anchors[self.xy[1], self.xy[2]]
        iou_tgt = []
        boxes_aligned = []
        anchors_aligned = []
        cls_tgt = []
        for b in range(batch_size):
            cur_anchors = batch_anchors[self.xy[0] == b].view(-1, 7)
            cur_boxes = gt_boxes[gt_boxes[:, 0] == box_indices[b], 1:]
            if len(cur_boxes) > 0 and len(cur_anchors) > 0:
                ious = boxes_iou_bev(cur_anchors, cur_boxes)
                ious_max, max_idx = ious.max(dim=1)
                pos = ious_max > self.iou_match
                neg = ious_max < self.iou_unmatch
                # down sample neg samples
                s = min(self.sample_size, pos.sum())
                if neg.sum() > self.sample_size:
                    perm = torch.randperm(neg.sum())[:self.sample_size]
                    neg = torch.where(neg)[0][perm]
                cls = torch.ones_like(ious_max) * -1
                cls[pos] = 1
                cls[neg] = 0
                boxes = cur_boxes[max_idx[pos]]
                anchors = cur_anchors[pos]
            else:
                cls = torch.ones_like(cur_anchors[:, 0]) * -1
                ious_max = torch.zeros_like(cur_anchors[:, 0])
                boxes = torch.empty((0, 7), device=cur_anchors.device)
                anchors = torch.empty((0, 7), device=cur_anchors.device)

            cls_tgt.append(cls.view(-1, 2))
            iou_tgt.append(ious_max.view(-1, 2))
            boxes_aligned.append(boxes)
            anchors_aligned.append(anchors)

        cls_tgt = torch.cat(cls_tgt, dim=0)
        iou_tgt = torch.cat(iou_tgt, dim=0)
        boxes_aligned = torch.cat(boxes_aligned, dim=0)
        anchors_aligned = torch.cat(anchors_aligned, dim=0)
        reg_tgt, dir_tgt = self.encode_boxes(anchors_aligned, boxes_aligned)

        return cls_tgt, iou_tgt, dir_tgt, reg_tgt

    def get_raw_points(self, batch_dict):
        raw_points = torch.cat([batch_dict['in_data'].C[:, :1],
                                batch_dict['xyz']], dim=-1)
        # raw_points = raw_points[batch_dict['target_semantic'] == 1]
        if len(raw_points[:, 0].unique()) > batch_dict['batch_size'] \
                and 'num_cav' in batch_dict:
            for i, c in enumerate(batch_dict['num_cav']):
                idx_start = sum(batch_dict['num_cav'][:i])
                mask = torch.logical_and(
                    raw_points[:, 0] >= idx_start,
                    raw_points[:, 0] < idx_start + c
                )
                raw_points[mask, 0] = i
        return raw_points

    def encode_boxes(self, anchors, boxes):
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(boxes, 1, dim=-1)

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha

        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)

        # encode box directions
        rgx = torch.cos(rg).view(-1, 1)                     # N 1
        rgy = torch.sin(rg).view(-1, 1)                     # N 1
        ra_ext = torch.cat([ra, ra + pi], dim=-1)           # N 2, invert
        rax = torch.cos(ra_ext)                             # N 2
        ray = torch.sin(ra_ext)                             # N 2
        diff_angle = torch.arccos(rax * rgx + ray * rgy)    # N 2
        dir_score = 1 - diff_angle / pi                     # N 2
        rtx = rgx - rax                                     # N 2
        rty = rgy - ray                                     # N 2

        dir_score = dir_score                               # N 2
        ret = [xt, yt, zt, wt, lt, ht, rtx, rty]
        reg = torch.cat(ret, dim=1)                         # N 6+4

        return reg, dir_score

    def decode_boxes(self):
        anchors = self.anchors[self.xy[1], self.xy[2]]
        boxes_enc = self.out['reg'].view(len(anchors), 2, 10)
        dir_scores = self.out['scores'][:, 2:].view(len(anchors), 2, 2)
        dir_scores = dir_scores.sigmoid()
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht = torch.split(boxes_enc[..., :6], 1, dim=-1)
        vt = boxes_enc[..., 6:]

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha

        ra_ext = torch.cat([ra, ra + pi], dim=-1)           # N 2, invert
        rax = torch.cos(ra_ext)                             # N 2
        ray = torch.sin(ra_ext)                             # N 2
        va = torch.cat([rax, ray], dim=-1)
        vg = vt + va
        rg = torch.atan2(vg[..., 2:], vg[..., :2]).view(-1, 2)

        dirs = torch.argmax(dir_scores, dim=-1).view(-1)
        rg = rg[torch.arange(len(rg)), dirs].view(len(xg), 2, 1)

        return torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)


class DetectionS2(nn.Module):
    def __init__(self, cfgs):
        super(DetectionS2, self).__init__()

        self.grid_size = 6
        grid = (self.grid_size, ) * 3 + (64,)
        self.grid_emb = nn.Parameter(torch.randn(grid))
        self.pos_emb_layer = linear_layers([3, 32, 32])
        self.attn_weight = linear_layers([64, 32, 1], ['ReLU', 'Sigmoid'])

        self.proj = linear_layers([64, 32])
        self.fc_layer = linear_layers([64, 64, 64])
        self.fc_out = linear_layers([64, 32])

        self.iou_head = nn.Linear(32, 1, bias=False)
        self.reg_head = nn.Linear(32, 8, bias=False)

        self.out = {}

    def forward(self, batch_dict):
        valid = batch_dict['target_semantic'] >= 0
        coords = self.get_coords(batch_dict)[valid]
        pred_boxes = batch_dict['detection']['boxes_fused']
        boxes = pred_boxes.clone()
        boxes[:, 4:7] *= 1.5
        boxes_decomposed, box_idxs_of_pts = points_in_boxes_gpu(
            coords, boxes, batch_dict['batch_size']
        )
        in_box_mask = box_idxs_of_pts >= 0
        new_idx = box_idxs_of_pts[in_box_mask]

        new_xyz = coords[in_box_mask, 1:]
        features = batch_dict['backbone']['p0'][valid][in_box_mask]
        mapped_boxes = boxes_decomposed[new_idx]

        # canonical transformation
        new_xyz = new_xyz - mapped_boxes[:, 1:4]
        xyz = new_xyz.clone()
        st = torch.sin(-mapped_boxes[:, -1])
        ct = torch.cos(-mapped_boxes[:, -1])
        new_xyz[:, 0] = xyz[:, 0] * ct - xyz[:, 1] * st
        new_xyz[:, 1] = xyz[:, 0] * st + xyz[:, 1] * ct

        # minus 1e-4 to ensure positive coords
        new_tfield = torch.div(new_xyz, mapped_boxes[:, 4:7]) * (6 - 1e-4) + 3

        new_tfield = ME.TensorField(
            coordinates=torch.cat([new_idx.view(-1, 1), new_tfield], dim=1),
            features=features,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
        )
        voxel_embs = self.voxelize_with_centroids(new_tfield, new_xyz)
        voxel_idx = voxel_embs.C[:, 0].long()
        num_box = len(boxes_decomposed)
        pos = voxel_embs.C.T[1:].long()
        # if pos.min() < 0 or pos.max() > self.grid_size - 1:
        #     print('d')
        pos_emb = self.grid_emb[pos[0], pos[1], pos[2]]
        voxel_embs = voxel_embs.F + pos_emb
        weights = self.attn_weight(voxel_embs)
        weighted_voxel_features = weights * voxel_embs
        out = torch.zeros_like(weighted_voxel_features[:num_box])
        torch_scatter.scatter_add(weighted_voxel_features,
                                  voxel_idx, dim=0, out=out)

        out = self.fc_out(out)
        ious = self.iou_head(out)
        regs = self.reg_head(out)

        box_src = boxes_decomposed
        box_src[:, 4:7] /= 1.5
        box_s2 = self.dec_box(box_src[:, 1:], regs)

        box_s2 = torch.cat([box_src[:, :1], box_s2], dim=-1)

        self.out = {
            'rois': box_src,
            'reg': regs,
            'iou': ious,
            'box': box_s2
        }
        return box_s2

    def get_coords(self, batch_dict):
        coords = torch.cat([batch_dict['in_data'].C[:, :1], batch_dict['xyz']], dim=-1)
        if len(coords[:, 0].unique()) > batch_dict['batch_size']:
            for i, c in enumerate(batch_dict['num_cav']):
                idx_start = sum(batch_dict['num_cav'][:i])
                mask = torch.logical_and(
                    coords[:, 0] >= idx_start,
                    coords[:, 0] < idx_start + c
                )
                coords[mask, 0] = i
        return coords

    def voxelize_with_centroids(self, x: ME.TensorField, coords: torch.Tensor):
        cm = x.coordinate_manager
        features = x.F
        # coords = x.C[:, 1:]

        out = x.sparse()
        size = torch.Size([len(out), len(x)])
        tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
        coords_p1, count_p1 = downsample_points(coords, tensor_map, field_map, size)
        norm_coords = normalize_points(coords, coords_p1, tensor_map)
        pos_emb = self.pos_emb_layer(norm_coords)
        feat_enc = self.proj(features)

        voxel_embs = self.fc_layer(torch.cat([feat_enc, pos_emb], dim=1))
        down_voxel_embs = downsample_embeddings(voxel_embs, tensor_map, size, mode="max")
        out = ME.SparseTensor(down_voxel_embs,
                              coordinate_map_key=out.coordinate_key,
                              coordinate_manager=cm)
        return out

    def loss(self, batch_dict):
        tgt_reg, tgt_iou, pos, src_iou = self.get_tgt(target_boxes=batch_dict['target_boxes'])
        if pos.sum() == 0:
            return 0, {}
        # selected = pos.float()
        # mask1 = src_iou > 0.7
        # mask2 = torch.logical_and(src_iou < 0.3, pos)
        loss_reg = weighted_smooth_l1_loss(self.out['reg'][pos], tgt_reg[pos])
        loss_iou = weighted_smooth_l1_loss(self.out['iou'].sigmoid(), tgt_iou)
        loss = 10 * loss_reg.mean() + loss_iou.mean()
        loss_dict = {'roi': loss}
        return loss, loss_dict

    def get_tgt(self, target_boxes):
        rois = self.out['rois']
        boxes = self.out['box']
        tgt_reg, tgt_boxes_aligned, pos_mask = self.enc_box(rois, target_boxes)
        tgt_iou = aligned_boxes_iou3d_gpu(boxes[:, 1:], tgt_boxes_aligned[:, 1:])
        src_iou = aligned_boxes_iou3d_gpu(rois[:, 1:], tgt_boxes_aligned[:, 1:])
        return tgt_reg, tgt_iou, pos_mask, src_iou

    @staticmethod
    @torch.no_grad()
    def enc_box(rois, gt_bbox):
        rois = rois.detach()
        ious = boxes_iou3d_gpu(rois[:, 1:], gt_bbox[:, 1:])

        ious_aligned = []
        boxes_aligned = []
        for i in rois[:, 0].unique():
            mask1 = rois[:, 0] == i
            mask2 = gt_bbox[:, 0] == i
            idx1, idx2 = torch.where(mask1)[0], torch.where(mask2)[0]
            cur_ious_max = ious[idx1.min(): idx1.max() + 1,
                           idx2.min(): idx2.max() + 1].max(dim=1)
            ious_aligned.append(cur_ious_max[0])
            boxes = gt_bbox[mask2][cur_ious_max[1]]
            boxes_aligned.append(boxes)

        boxes_aligned = torch.cat(boxes_aligned, dim=0)
        ious_aligned = torch.cat(ious_aligned, dim=0)
        valid = ious_aligned > 0.1

        xa, ya, za, wa, la, ha, ra = torch.split(rois[valid, 1:], 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(boxes_aligned[valid, 1:], 1, dim=-1)

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha

        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)

        ct = torch.cos(rg) - torch.cos(ra)
        st = torch.sin(rg) - torch.sin(ra)

        ret = torch.cat([xt, yt, zt, wt, lt, ht, ct, st], dim=-1)
        reg_boxes = torch.zeros_like(rois)
        reg_boxes[valid] = ret
        return reg_boxes, boxes_aligned, valid

    @staticmethod
    @torch.no_grad()
    def dec_box(anchors, reg):
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, ct, st = torch.split(reg, 1, dim=-1)

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha

        cg = ct + torch.cos(ra)
        sg = st + torch.sin(ra)
        rg = torch.atan2(sg, cg)

        ret = torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)
        return ret


