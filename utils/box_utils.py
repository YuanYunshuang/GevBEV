import numpy as np
import torch

from ops.utils import points_in_boxes_cpu, points_in_boxes_gpu
from utils.misc import pad_list_to_array_torch


def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - np.floor(val / period + offset) * period


def encode_boxes_relative(points, boxes, lwh_mean, labels=None):
    """

    :param points: list of (N, 3) np.ndarray
    :param labels: list of (N,) np.ndarray
    :param boxes: list of (M, 7) np.ndarray
    :param lwh_mean: list or np.ndarray of length 3
    :return:
    """
    lwh_mean = np.array(lwh_mean).reshape(1, 3)
    if labels is not None:
        masks = [ls == 1 for ls in labels] # cls vehicle
        points_masked = [ps[m, :3] for ps, m in zip(points, masks)]
    pt_to_box_list = []
    point_indices_list = []
    points_in_box_list = []
    for pts, bbx in zip(points_masked, boxes):
        point_indices = points_in_boxes_cpu(pts, bbx) # BxN
        point_indices, box_indices = np.where(point_indices.T)
        pt_to_box_list.append(bbx[box_indices])
        point_indices_list.append(point_indices)
        points_in_box_list.append(pts[point_indices])

    boxes_cat = np.concatenate(pt_to_box_list, axis=0)
    points_in_box = np.concatenate(points_in_box_list, axis=0)
    points_cat = np.concatenate(points, axis=0)

    tgt = np.zeros((len(points_cat), 8))
    xyz_t = (points_in_box - boxes_cat[:, :3]) / lwh_mean
    lwh_t = boxes_cat[:, 3:6] / lwh_mean
    rs_t = np.sin(boxes_cat[:, 6:])
    rc_t = np.cos(boxes_cat[:, 6:])

    cnts = [len(l) for l in points_masked]
    point_indices_list = [idxs + sum(cnts[:i]) for i, idxs in enumerate(point_indices_list)]
    point_indices_cat = np.concatenate(point_indices_list, axis=0)
    point_mask_cat = np.concatenate(masks, axis=0)
    tgt[np.where(point_mask_cat)[0][point_indices_cat]] = np.concatenate([xyz_t, lwh_t, rs_t, rc_t], axis=1)

    return tgt


def decode_boxes(reg, points, lwh_mean):
    assert len(reg)==len(points)
    if not isinstance(lwh_mean, torch.Tensor):
        lwh_mean = torch.Tensor(lwh_mean).view(1, 3)
    points = points.to(reg.device)
    lwh_mean = lwh_mean.to(reg.device)

    diagonal = torch.norm(lwh_mean[0, :2])
    # encode with diagonal length
    xy = reg[:, :2] * diagonal + points[:, :2]
    z = reg[:, 2:3] * lwh_mean[0, 2] + points[:, 2:3]
    lwh = reg[:, 3:6].exp() * lwh_mean
    r = torch.atan2(reg[:, 6:7], reg[:, 7:])

    return torch.cat([xy, z, lwh, r], dim=-1)


def decode_boxes_relative(reg, points, lwh_mean):
    assert len(reg)==len(points)
    if not isinstance(lwh_mean, torch.Tensor):
        lwh_mean = torch.Tensor(lwh_mean).view(1, 3)
    points = points.to(reg.device)
    lwh_mean = lwh_mean.to(reg.device)

    xyz = points - reg[:, :3] * lwh_mean
    lwh = reg[:, 3:6] * lwh_mean + lwh_mean
    r = torch.atan2(reg[:, 6:7], reg[:, 7:])

    return torch.cat([xyz, lwh, r], dim=-1)


def encode_boxes_relative_torch(points, boxes, lwh_mean):
    points_batch = pad_list_to_array_torch(points)
    boxes_batch = pad_list_to_array_torch(boxes)
    dtype = points_batch.dtype
    device = points_batch.device
    lwh_mean = torch.tensor(lwh_mean, dtype=dtype, device=device).view(1, 3)
    box_idxs_of_pts = points_in_boxes_gpu(points_batch, boxes_batch)
    mask = box_idxs_of_pts > 0
    points_pos = points_batch[mask]
    tgt_batch = torch.zeros(tuple(points_batch.shape[:-1]) + (8,),
                      dtype=dtype, device=device)
    boxes_pos = boxes_batch[torch.where(mask)[0], box_idxs_of_pts[mask].long()]

    xyz_t = (points_pos - boxes_pos[:, :3]) / lwh_mean
    lwh_t = boxes_pos[:, 3:6] / lwh_mean
    rs_t = torch.sin(boxes_pos[:, 6:])
    rc_t = torch.cos(boxes_pos[:, 6:])
    tgt_batch[mask] = torch.cat([xyz_t, lwh_t, rs_t, rc_t], dim=1)
    tgt_reg = []
    tgt_cls = []
    for c, t, p in zip(mask.int(), tgt_batch, points):
        tgt_reg.append(t[:len(p)])
        tgt_cls.append(c[:len(p)])

    return tgt_cls, tgt_reg


def boxes_to_corners_2d(boxes_np):
    """
    Convert boxes to 4 corners in xy plane
    :param boxes_np: np.ndarray [N, 7], cols - (x,y,z,dx,dy,dz,r)
    :return: corners: np.ndarray [N, 4, 2], corner order is
    back left, front left, front back, back left
    """
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = - dx / 2
    y1 = - dy / 2
    x2 = + dx / 2
    y2 = + dy / 2
    theta = boxes_np[:, 6:7]
    # bl, fl, fr, br
    corners = np.array([[x1, y2],[x2,y2], [x2,y1], [x1, y1]]).transpose(2, 0, 1)
    new_x = corners[:, :, 0] * np.cos(theta) + \
            corners[:, :, 1] * -np.sin(theta) + x[:, None]
    new_y = corners[:, :, 0] * np.sin(theta) + \
            corners[:, :, 1] * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)

    return corners
