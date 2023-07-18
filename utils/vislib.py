import sys
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

use_wandb = True


def visualization(func_list, batch_data):
    for func_str in func_list:
        getattr(sys.modules[__name__], func_str)(batch_data)


def log_loss(batch_data):
    loss_dict = batch_data['loss_dict']
    wandb.log(loss_dict)


def draw_box_plt(boxes_dec, ax, color=None, linewidth_scale=1.0, linestyle='solid'):
    """
    draw boxes in a given plt ax
    :param boxes_dec: (N, 5) or (N, 7) in metric
    :param ax:
    :return: ax with drawn boxes
    """
    if not len(boxes_dec)>0:
        return ax
    boxes_np= boxes_dec
    if isinstance(boxes_np, torch.Tensor):
        boxes_np = boxes_np.cpu().detach().numpy()
    elif isinstance(boxes_np, list):
        boxes_np = np.array(boxes_np)
    if boxes_np.shape[-1]>5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:5]
    # bl, fl, fr, br
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1]
              - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1]
              - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)
    for corner in corners:
        ax.plot(corner[[0,1,2,3,0], 0], corner[[0,1,2,3,0], 1], color=color,
                linewidth=linewidth_scale, linestyle=linestyle)
        # draw direction
        # front = corner[[2, 3]].mean(axis=0)
        # center = corner.mean(axis=0)
        # ax.plot([front[0], center[0]], [front[1], center[1]], color=color,
        #         linewidth=linewidth_scale)
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=1.5*linewidth_scale)
    return ax


def draw_points_boxes_plt(pc_range=None, points=None, boxes_pred=None, boxes_gt=None, wandb_name=None,
                          points_c='k', bbox_gt_c='green', bbox_pred_c='red',
                          return_ax=False, ax=None, marker_size=2.0):
    if pc_range is not None:
        if isinstance(pc_range, int) or isinstance(pc_range, int):
            pc_range = [-pc_range, -pc_range, pc_range, pc_range]
        elif isinstance(pc_range, list) and len(pc_range)==6:
            pc_range = [pc_range[i] for i in [0, 1, 3, 4]]
        else:
            assert isinstance(pc_range, list) and len(pc_range)==4, \
                "pc_range should be a int, float or list of lenth 6 or 4"
    if ax is None:
        ax = plt.figure(figsize=(pc_range[2] / 10, pc_range[3] / 10)).add_subplot(1, 1, 1)
        ax.set_aspect('equal', 'box')
    if pc_range is not None:
        ax.set(xlim=(pc_range[0], pc_range[2]),
               ylim=(pc_range[1], pc_range[3]))
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], '.',
                color=points_c, markersize=marker_size)
    if (boxes_pred is not None) and len(boxes_pred) > 0:
        ax = draw_box_plt(boxes_pred, ax, color=bbox_pred_c)
    if (boxes_gt is not None) and len(boxes_gt) > 0:
        ax = draw_box_plt(boxes_gt, ax, color=bbox_gt_c)
    plt.xlabel('x')
    plt.ylabel('y')

    if wandb_name is not None:
        wandb.log({wandb_name: wandb.Image(plt)})
    if return_ax:
        return ax
