import functools
import torch, torch_scatter
import MinkowskiEngine as ME
from torch import nn
from model.submodules.utils import minkconv_conv_block, indices2metric, \
    pad_r, linear_last, fuse_batch_indices, meshgrid, metric2indices, \
    weighted_mahalanobis_dists
from model.submodules.bev_base import BEVBase
from model.losses.edl import edl_mse_loss


class HEviBev(nn.Module):
    def __init__(self, cfgs):
        super(HEviBev, self).__init__()
        self.heads = []
        for cfg in cfgs:
            k = list(cfg.keys())[0]
            setattr(self, k, EviBEV(cfg[k]))
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


class EviBEV(BEVBase):
    def __init__(self, cfgs):
        super(EviBEV, self).__init__(cfgs)

    def get_reg_layer(self, in_dim):
        return linear_last(in_dim, 32, 2, bias=True)

    def draw_distribution(self, reg):
        reg_evi = reg.relu()
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
        return evidence

    def loss(self, batch_dict):
        tgt, indices, _ = self.get_tgt(batch_dict)
        evidence = self.out['evidence'][
            indices[0], indices[1], indices[2]
        ]
        epoch_num = batch_dict.get('epoch', 0)
        loss, loss_dict = edl_mse_loss(self.name[:3], evidence, tgt,
                                       epoch_num, 2, self.annealing_step)
        # we boost var with a small weighted loss to encourage larger vars
        # if epoch_num > 40:
        #     loss_boost = torch.exp(-self.out['reg'].relu()).mean()
        #     loss = loss + loss_boost * 0.01
        return loss, loss_dict

