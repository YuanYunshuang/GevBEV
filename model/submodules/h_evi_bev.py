import torch
from model.submodules.utils import metric2indices, linear_last
from model.submodules.bev_base import BEVBase, HBEVBase
from model.losses.edl import edl_mse_loss


class EviBEV(BEVBase):
    def __init__(self, cfgs):
        super(EviBEV, self).__init__(cfgs)

    def get_reg_layer(self, in_dim):
        return linear_last(in_dim, 32, 2, bias=True)

    def draw_distribution(self, batch_dict):
        reg = self.out['reg']
        reg_evi = reg.relu()
        ctrs = self.centers[:, :3]  # N 2
        batch_size = ctrs[:, 0].max().int() + 1
        evidence = torch.zeros((batch_size, self.size_x, self.size_y, 2),
                               device=reg_evi.device)
        inds = metric2indices(ctrs, self.res).T
        inds[1] -= self.offset_sz_x
        inds[2] -= self.offset_sz_y
        # obs_mask = evidence[..., 0].bool()
        # obs_mask[inds[0], inds[1], inds[2]] = True
        evidence[inds[0], inds[1], inds[2]] = reg_evi
        self.out['evidence'] = evidence
        return evidence

    def loss(self, batch_dict):
        tgt_pts, tgt_label, indices = self.get_tgt(batch_dict, discrete=True)
        evidence_map = self.draw_distribution(self.out['reg'])
        evidence = evidence_map[
            indices[0], indices[1], indices[2]
        ]
        epoch_num = batch_dict.get('epoch', 0)
        loss, loss_dict = edl_mse_loss(self.name[:3], evidence, tgt_label,
                                       epoch_num, 2, self.annealing_step)
        return loss, loss_dict


class HEviBev(HBEVBase):
    DISTR_CLS = EviBEV
    def __init__(self, cfgs):
        super(HEviBev, self).__init__(cfgs)

