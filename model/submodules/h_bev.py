import functools
import torch, torch_scatter
import MinkowskiEngine as ME
from torch import nn
from model.submodules.utils import minkconv_conv_block, indices2metric, \
    pad_r, linear_last, fuse_batch_indices, meshgrid, metric2indices
from model.submodules.bev_base import BEVBase
from ops.utils import points_in_boxes_gpu
from model.losses.common import cross_entroy_with_logits


class HBev(nn.Module):
    def __init__(self, cfgs):
        super(HBev, self).__init__()
        self.heads = []
        for cfg in cfgs:
            k = list(cfg.keys())[0]
            setattr(self, k, BEV(cfg[k]))
            self.heads.append(k)

    def forward(self, batch_dict):
        for h in self.heads:
            getattr(self, h)(batch_dict)

    def loss(self, batch_dict):
        loss = 0
        loss_dict = {}
        for h in self.heads:
            l, ldict = getattr(self, h).loss(batch_dict)
            loss = loss + l
            loss_dict.update(ldict)
        return loss, loss_dict


class BEV(BEVBase):
    def __init__(self, cfgs):
        super(BEV, self).__init__(cfgs)

    def get_reg_layer(self, in_dim):
        return linear_last(in_dim, 32, 2, bias=True)

    def draw_distribution(self, reg):
        reg_evi = reg
        ctrs = self.centers[:, :3]  # N 2
        batch_size = ctrs[:, 0].max().int() + 1
        evidence = torch.zeros((batch_size, self.size * 2, self.size * 2, 2),
                               device=reg_evi.device)
        inds = metric2indices(ctrs, self.res).T
        inds[1:] += self.size
        # obs_mask = evidence[..., 0].bool()
        # obs_mask[inds[0], inds[1], inds[2]] = True
        evidence[inds[0], inds[1], inds[2]] = reg_evi
        self.out['evidence'] = evidence
        return evidence.softmax(dim=-1)

    def loss(self, batch_dict):
        tgt, indices, _ = self.get_tgt(batch_dict)
        preds = self.out['evidence'][
            indices[0], indices[1], indices[2]
        ]
        loss = cross_entroy_with_logits(preds, tgt, 2, reduction='mean')
        ss = preds.detach()
        tt = tgt.detach()
        acc = (torch.argmax(ss, dim=1) == tt).sum() / len(tt) * 100
        loss_dict = {
            f'{self.name[:3]}': loss,
            f'{self.name[:3]}_ac': acc,
        }
        return loss, loss_dict

