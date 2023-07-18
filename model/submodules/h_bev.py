import functools
import torch, torch_scatter
import MinkowskiEngine as ME
from torch import nn
from model.submodules.utils import minkconv_conv_block, indices2metric, \
    pad_r, linear_last, fuse_batch_indices, meshgrid, metric2indices
from model.submodules.bev_base import BEVBase, HBEVBase
from ops.utils import points_in_boxes_gpu
from model.losses.common import cross_entroy_with_logits


class BEV(BEVBase):
    def __init__(self, cfgs):
        super(BEV, self).__init__(cfgs)

    def get_reg_layer(self, in_dim):
        return linear_last(in_dim, 32, 2, bias=True)

    def draw_distribution(self, batch_dict):
        """
        Convert sparse regression logits into to sense logits map, and return softmax result of logits.
        """
        reg = self.out['reg']
        reg_evi = reg
        ctrs = self.centers[:, :3]  # N 3
        batch_size = ctrs[:, 0].max().int() + 1
        evidence = torch.zeros((batch_size, self.size_x, self.size_y, 2),
                               device=reg_evi.device)
        inds = metric2indices(ctrs, self.res).T
        inds[1] -= self.offset_sz_x
        inds[2] -= self.offset_sz_y

        evidence[inds[0], inds[1], inds[2]] = reg_evi
        self.out['evidence'] = evidence
        return evidence.softmax(dim=-1)

    def loss(self, batch_dict):
        tgt_pts, tgt_label, indices = self.get_tgt(batch_dict, discrete=True)
        self.draw_distribution(self.out['reg'])
        preds_map = self.out['evidence']

        preds = preds_map[indices[0], indices[1], indices[2]]
        loss = cross_entroy_with_logits(preds, tgt_label, 2, reduction='mean')
        ss = preds.detach()
        tt = tgt_label.detach()
        acc = (torch.argmax(ss, dim=1) == tt).sum() / len(tt) * 100
        loss_dict = {
            f'{self.name[:3]}': loss,
            f'{self.name[:3]}_ac': acc,
        }
        return loss, loss_dict


class HBev(HBEVBase):
    DISTR_CLS = BEV
    def __init__(self, cfgs):
        super(HBev, self).__init__(cfgs)

