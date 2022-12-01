import torch
import torch.nn.functional as F


def weighted_smooth_l1_loss(preds, targets, sigma=3.0, weights=None):
    diff = preds - targets
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (sigma ** 2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * sigma, 2) + \
               (abs_diff - 0.5 / (sigma ** 2)) * (1.0 - abs_diff_lt_1)
    if weights is not None:
        if len(loss.shape) > len(weights.shape):
            weights = weights.unsqueeze(dim=-1)
        loss *= weights
    return loss


def sigmoid_binary_cross_entropy(preds, tgts, weights=None, reduction='none'):
    if weights is not None and len(preds.shape)>len(weights.shape):
        weights = weights.unsqueeze(-1)
    per_entry_cross_ent = F.binary_cross_entropy_with_logits(preds.view(-1), tgts.view(-1),
                                                             weights, reduction=reduction)
    return per_entry_cross_ent


def cross_entroy_with_logits(preds, tgts, n_cls, weights=None, reduction='none'):
    cared = tgts >= 0
    preds = preds[cared]
    tgts = tgts[cared]
    tgt_onehot = torch.zeros((len(tgts), n_cls), device=preds.device)
    tgt_onehot[torch.arange(len(tgts), device=tgts.device), tgts.long()] = 1

    loss = F.cross_entropy(preds, tgt_onehot, weight=weights, reduction=reduction)
    return loss