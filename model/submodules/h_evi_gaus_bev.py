import torch, torch_scatter
from model.submodules.utils import draw_sample_prob, \
    linear_last, meshgrid, metric2indices, \
    weighted_mahalanobis_dists
from model.submodules.bev_base import BEVBase, HBEVBase
from model.losses.edl import edl_mse_loss
import matplotlib.cm as cm
cm_hot = cm.get_cmap('hot')

class EviGausBEV(BEVBase):
    def __init__(self, cfgs):
        super(EviGausBEV, self).__init__(cfgs)
        res = self.stride * self.voxel_size
        steps = int(self.distr_r / res) * 2 + 1
        offset = meshgrid(-self.distr_r, self.distr_r, 2,
                          n_steps=steps).cuda().view(-1, 2)
        self.nbrs = offset[torch.norm(offset, dim=1) < 2].view(1, -1, 2)

    def get_reg_layer(self, in_dim):
        return linear_last(in_dim, 32, 6, bias=True)

    def draw_distribution(self, batch_dict):
        reg = self.out['reg'].relu()
        reg_evi = reg[:, :2]
        reg_var = reg[:, 2:].view(-1, 2, 2)
        ctrs = self.centers[:, :3]  # N 2

        dists = torch.zeros_like(ctrs[:, 1:].view(-1, 1, 2)) + self.nbrs
        probs_weighted = weighted_mahalanobis_dists(reg_evi, reg_var, dists, self.var0)
        evidence, obs_mask = self.get_evidence_map(probs_weighted, ctrs)

        if getattr(self, 'cpm_option', 'none') == 'none' or self.training:
            evidence_fused = evidence
            Nall, Nsel = None, None
        else:
            unc, conf = self.evidence_to_conf_unc(evidence)
            if getattr(self, 'cpm_option', 'none') == 'road':
                assert batch_dict['bevmap_static'] is not None, 'gt road bev-opv2v is not available.'
                sx, sy = batch_dict['bevmap_static'].shape[1:3]
                sx = sx // self.size_x
                sy = sy // self.size_y
                cared_mask = batch_dict['bevmap_static'][:, ::sx, ::sy].bool()
            elif getattr(self, 'cpm_option', 'none') == 'all':
                cared_mask = batch_dict['distr_conv_out'][f'p{self.stride}']['obs_mask']
                cared_mask = torch.ones_like(cared_mask).bool()
            else:
                raise NotImplementedError

            evidence_cpm, Nall, Nsel = self.get_cpm_evimap(
                batch_dict['num_cav'], unc, evidence, cared_mask, batch_dict
            )

            ego_idx = [sum(batch_dict['num_cav'][:i]) for i in range(len(batch_dict['num_cav']))]
            evidence_ego = evidence[ego_idx]
            evidence_fused = evidence_ego + evidence_cpm

        self.out['evidence'] = evidence_fused
        self.out['Nall'] = Nall
        self.out['Nsel'] = Nsel
        return evidence_fused

    def get_evidence_map(self, probs_weighted, coor):
        voxel_new = coor[:, 1:].view(-1, 1, 2) + self.nbrs
        # convert metric voxel points to map indices
        x = (torch.floor(voxel_new[..., 0] / self.res) - self.offset_sz_x).long()
        y = (torch.floor(voxel_new[..., 1] / self.res) - self.offset_sz_y).long()
        batch_indices = (torch.ones_like(probs_weighted[:, :, 0]) * coor[:, :1]).long()
        mask = (x >= 0) & (x < self.size_x) & (y >= 0) & (y < self.size_y)
        x, y = x[mask], y[mask]
        batch_indices = batch_indices[mask]

        # copy sparse probs to the dense evidence map
        indices = batch_indices * self.size_x * self.size_y + x * self.size_y + y
        batch_size = coor[:, 0].max().int().item() + 1
        probs_weighted = probs_weighted[mask].view(-1, 2)
        evidence = torch.zeros((batch_size, self.size_x, self.size_y, 2),
                               device=probs_weighted.device).view(-1, 2)
        torch_scatter.scatter(probs_weighted, indices,
                              dim=0, out=evidence, reduce='sum')
        evidence = evidence.view(batch_size, self.size_x, self.size_y, 2)

        # create observation mask
        obs_mask = torch.zeros_like(evidence[..., 0]).view(-1)
        obs = indices.unique().long()
        obs_mask[obs] = 1
        obs_mask = obs_mask.view(batch_size, self.size_x, self.size_y).bool()
        return evidence, obs_mask

    def get_cpm_evimap(self, num_cav, unc, evidence, cared_mask, batch_dict=None):
        """
        1. Count pixels of sharing all observed and masked (i.e road) areas
        2. Count pixels of sharing selected area which is filter by a given unc. threshold
        3. Build the CPM after selection.
        """
        evi_share = []
        n_share_all = 0
        n_share_selected = 0
        for i, c in enumerate(num_cav):
            if c == 1:
                evi_share.append(torch.zeros_like(evidence[0]))
                continue
            # get unc. and evidence map for the current batch
            idx_start = sum(num_cav[:i])
            cur_unc = unc[idx_start:idx_start + c]
            cur_evi = evidence[idx_start:idx_start + c]

            # share all info on masked area
            resp_all = torch.logical_and(cur_unc[1:] < 1.0, cared_mask[i].unsqueeze(0))
            n_share_all = n_share_all + resp_all.sum().item()
            req_mask = torch.logical_and(cur_unc[0] >= self.cpm_thr, cared_mask[i])

            # coop mask for responding
            rsp_mask = torch.logical_and(cur_unc[1:] < 1.0, req_mask.unsqueeze(0))

            # share selected
            n_share_selected = n_share_selected + rsp_mask.sum().item()
            # get evidence map of coop. CAV
            evi_coop = cur_evi[1:].clone()
            evi_coop[torch.logical_not(rsp_mask)] = 0
            evi_share.append(evi_coop.sum(dim=0))
            
        evi_share = torch.stack(evi_share, dim=0)

        n_share_all = n_share_all / len(num_cav)
        n_share_selected = n_share_selected / len(num_cav)
        return evi_share, n_share_all, n_share_selected

    def get_cpm_centerpoints(self, num_cav, unc, centers, reg, road_bev, thr=0.5):
        centers_share = []
        reg_share = []
        for i, c in enumerate(num_cav):
            if c == 1:
                continue
            idx_start = sum(num_cav[:i])
            cur_unc = unc[idx_start:idx_start + c]
            # ego mask for requesting the CPM from coop. CAV
            req_mask = torch.logical_and(cur_unc[0] > thr, road_bev[i])
            # coop mask for responding
            rsp_mask = torch.logical_and(cur_unc[1:] < thr, req_mask.unsqueeze(0))
            # get the center points of coop. CAV
            mask = torch.logical_and(centers[:, 0] >= idx_start,
                                     centers[:, 0] < idx_start + c)
            cur_ctrs = centers[mask]
            cur_reg = reg[mask]
            cur_ctrs[:, 0] = cur_ctrs[:, 0] - idx_start
            cur_ctrs[:, 1:] = cur_ctrs[:, 1:] - self.voxel_size / 2
            res = self.voxel_size * self.stride
            grid_size = int(self.det_r / res * 2)
            indices = metric2indices(cur_ctrs, res)
            indices[:, 1:] = indices[:, 1:] + grid_size / 2
            mask = torch.logical_and(indices[:, 1:] >= 0, indices[:, 1:] < grid_size).all(dim=1)
            indices = indices[mask]
            cur_ctrs = cur_ctrs[mask]
            cur_reg = cur_reg[mask]
            ctr_mask = rsp_mask[indices[:, 0] - 1, indices[:, 1], indices[:, 2]]
            cur_ctrs = cur_ctrs[ctr_mask]
            cur_ctrs[:, 0] = i
            centers_share.append(cur_ctrs)
            reg_share.append(cur_reg[ctr_mask])
        centers_share = torch.cat(centers_share, dim=0)
        reg_share = torch.cat(reg_share, dim=0)

        return centers_share, reg_share

    def evidence_to_conf_unc(self, evidence):
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        conf = torch.div(alpha, S)
        K = evidence.shape[-1]
        unc = torch.div(K, S)
        # conf = torch.sqrt(conf * (1 - unc))
        unc = unc.squeeze(dim=-1)
        return unc, conf

    def loss(self, batch_dict):
        tgt_pts, tgt_labels, indices = self.get_tgt(batch_dict)

        evidence = draw_sample_prob(self.centers[:, :3],
                                    self.out['reg'].relu(),
                                    tgt_pts,
                                    self.res,
                                    self.distr_r,
                                    self.lidar_range,
                                    batch_dict['batch_size'],
                                    var0=self.var0)
        epoch_num = batch_dict.get('epoch', 0)
        loss, loss_dict = edl_mse_loss(self.name[:3], evidence, tgt_labels,
                                       epoch_num, 2, self.annealing_step)
        return loss, loss_dict


class HEviGausBev(HBEVBase):
    DISTR_CLS = EviGausBEV
    def __init__(self, cfgs):
        super(HEviGausBev, self).__init__(cfgs)

