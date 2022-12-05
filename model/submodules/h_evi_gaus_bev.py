import functools
import torch, torch_scatter
import MinkowskiEngine as ME
from torch import nn
from model.submodules.utils import draw_sample_prob, \
    pad_r, linear_last, indices2metric, meshgrid, metric2indices, \
    weighted_mahalanobis_dists
from model.submodules.bev_base import BEVBase
from model.losses.edl import edl_mse_loss


class HEviGausBev(nn.Module):
    def __init__(self, cfgs):
        super(HEviGausBev, self).__init__()
        self.heads = []
        for cfg in cfgs:
            k = list(cfg.keys())[0]
            setattr(self, k, EviGausBEV(cfg[k]))
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


class EviGausBEV(BEVBase):
    def __init__(self, cfgs):
        super(EviGausBEV, self).__init__(cfgs)

    def get_reg_layer(self, in_dim):
        return linear_last(in_dim, 32, 6, bias=True)

    def draw_distribution(self, reg):
        if not self.training or not self.sample_pixels:
            reg = reg.relu()
            reg_evi = reg[:, :2]
            reg_var = reg[:, 2:].view(-1, 2, 2)
            ctrs = self.centers[:, :3]  # N 2

            dists = torch.zeros_like(ctrs[:, 1:].view(-1, 1, 2)) + self.nbrs
            probs_weighted = weighted_mahalanobis_dists(reg_evi, reg_var, dists, self.var0)
            evidence, obs_mask = self.get_evidence_map(probs_weighted, ctrs)
        else:
            evidence = None
        self.out['evidence'] = evidence
        return evidence

    def get_evidence_map(self, probs_weighted, coor):
        voxel_new = coor[:, 1:].view(-1, 1, 2) + self.nbrs
        size = self.size
        xy = (torch.floor(voxel_new / self.res) + size).view(-1, 2)
        mask = torch.logical_and(xy >= 0, xy < (size * 2)).all(dim=1)
        xy = xy[mask]
        batch_indices = (torch.ones_like(probs_weighted[:, :, 0]) * coor[:, :1]).view(-1)
        batch_indices = batch_indices[mask]
        indices = batch_indices * (size * 2) ** 2 + xy[:, 0] * (size * 2) + xy[:, 1]
        batch_size = coor[:, 0].max().int() + 1
        probs_weighted = probs_weighted.view(-1, 2)[mask]
        evidence = torch.zeros((batch_size, size * 2, size * 2, 2),
                               device=probs_weighted.device).view(-1, 2)
        torch_scatter.scatter(probs_weighted, indices.long(),
                              dim=0, out=evidence, reduce='sum')
        evidence = evidence.view(batch_size, size * 2, size * 2, 2)
        obs_mask = torch.zeros_like(evidence[..., 0]).view(-1)
        obs = indices.unique().long()
        obs_mask[obs] = 1
        obs_mask = obs_mask.view(batch_size, size * 2, size * 2).bool()
        return evidence, obs_mask

    def loss(self, batch_dict):
        tgt, indices, bev_pts = self.get_tgt(batch_dict)
        if self.out['evidence'] is not None:
            evidence = self.out['evidence'][indices[0], indices[1], indices[2]]
        else:
            evidence = draw_sample_prob(self.centers[:, :3],
                                        self.out['reg'].relu(),
                                        bev_pts, self.res, self.distr_r, self.det_r,
                                        batch_dict['batch_size'],
                                        var0=self.var0)
        epoch_num = batch_dict.get('epoch', 0)
        loss, loss_dict = edl_mse_loss(self.name[:3], evidence, tgt,
                                       epoch_num, 2, self.annealing_step)
        # we boost var with a small weighted loss to encourage larger vars
        # if epoch_num > 40:
        #     loss_boost = torch.exp(-self.out['reg'].relu()).mean()
        #     loss = loss + loss_boost * 0.01
        return loss, loss_dict

