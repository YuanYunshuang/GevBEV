import glob
import json

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import cv2
import tqdm

from tools.eval_bev import parse_evibev_inf
from ops.utils import points_in_boxes_gpu

log_dirs = {'evigausbev-opv2v': 'GevBEV', 'evibev-opv2v': 'EviBEV', 'bev-opv2v': 'BEV'}
log_path = '/mars/projects20/evibev_exp/opv2v'


def evidence_to_conf_unc(evidence, is_edl):
    if is_edl:
        # used edl loss
        alpha = evidence + 1
        S = torch.sum(alpha, dim=-1, keepdim=True)
        conf = torch.div(alpha, S)
        K = evidence.shape[-1]
        unc = torch.div(K, S)
        # conf = torch.sqrt(conf * (1 - unc))
        unc = unc.squeeze(dim=-1)
    else:
        # use entropy as uncertainty
        entropy = -evidence * torch.log2(evidence)
        unc = entropy.sum(dim=-1)
        # conf = torch.sqrt(evidence * (1 - unc.unsqueeze(-1)))
        conf = evidence
    return conf, unc


def generate_gt_bev(gt_boxes, batch_size, grid_size, res, lidar_range):
    sx, sy = grid_size
    gt_mask = torch.ones((batch_size, sx, sy), device=gt_boxes.device)
    if len(gt_boxes) > 0:
        indices = torch.stack(torch.where(gt_mask), dim=1)
        ixy = indices.float()
        ixy[:, 1:] = (ixy[:, 1:] + 0.5) * res
        ixy[:, 1] += lidar_range[0]
        ixy[:, 2] += lidar_range[1]
        ixyz = F.pad(ixy, (0, 1), 'constant', 0.0)
        boxes = gt_boxes.clone()
        boxes[:, 3] = 0
        boxes_decomposed, box_idx_of_pts = points_in_boxes_gpu(
            ixyz, boxes, batch_size=batch_size
        )
        inds = indices[box_idx_of_pts >= 0].T
        gt_mask[inds[0], inds[1], inds[2]] = 0
    gt_mask_all = torch.logical_not(gt_mask)
    return gt_mask_all



def prc(head_name):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    for log_dir, name in log_dirs.items():
        data = torch.load(os.path.join(log_path, f'{log_dir}_p2', 'test',
                                       f'{head_name}_plt_data.pth'))
        pr = data['pr_curve']
        ax.plot(pr[:, 0], pr[:, 1], label=name)
        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 1])
    ax.legend()
    plt.savefig(os.path.join(log_path, f'prc_{head_name}.png'))
    plt.close()


def unc_q(head_name):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    thrs = np.arange(0, 1, 0.1)
    if head_name == 'surface':
        colors = ['#AAC8A7', '#E9FFC2', '#FDFFAE']
    else:
        colors = ['#FAAB78', '#FFD495', '#FFFBAC']
    for i, log_dir in enumerate(log_dirs):
        data = torch.load(os.path.join(log_path, log_dir,
                                       'test0-0', f'{head_name}_plt_data.pth'))
        unc_q = data['unc_Q']
        # ax.plot(thrs + 0.05, unc_q, label=log_dir)
        for thr, r in zip(thrs, unc_q):
            bar = ax.bar(thr + i * 0.02 + 0.02, [r], width=0.02,
                    bottom=[0],
                    color=[colors[i]],
                    edgecolor ='k',
                    joinstyle='round')
            if thr==0:
                bar.set_label(log_dirs[log_dir])
    ax.plot([0, 1], [1, 0.5], '--k')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0.5, 1])
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Weighted Accuracy')
    plt.savefig(os.path.join(log_path, f'unc_q_{head_name}.png'))
    plt.close()


def cpm(head='obj'):
    hi = 0 if head == 'sur' else 1
    nall = [45783.4, 46698.0]
    nobs = [43466.9, 44344.7]
    nroad = [16271.6, 16748.1]
    baselines_sur = [67.4, 70.3]
    baselines_obj = [74.5, 74.8]
    thrs = np.arange(0.1, 1, 0.1)
    colors = ['blue', 'green']
    fig = plt.figure(figsize=(9, 3))
    axs = fig.subplots(1, 3)
    for i, m in enumerate(['all', 'road']):
        cpm_all = []
        iou_all = []
        iou_obs = []
        for thr in range(1, 10):
            cur_path = f"{log_path}/evigausbev_cpm_{m}/test{thr}/result.txt"
            with open(cur_path, 'r') as fh:
                lines = fh.readlines()
                nsur = float(lines[2].strip().split(':')[-1]) * 8 / 1024
                iou_sur_all = float(lines[4].strip().split(' ')[-1])
                iou_sur_obs = float(lines[5].strip().split(' ')[-1])
                nobj = float(lines[8].strip().split(':')[-1]) * 8 / 1024
                iou_obj_all = float(lines[10].strip().split(' ')[-1])
                iou_obj_obs = float(lines[11].strip().split(' ')[-1])
                cpm_all.append(locals().get(f'n{head}'))
                iou_all.append(locals().get(f'iou_{head}_all'))
                iou_obs.append(locals().get(f'iou_{head}_obs'))

        axs[0].plot(thrs, cpm_all, '*-', color=colors[i], label=m)
        axs[0].plot([0.1, 0.9], [locals().get(f'n{m}')[hi] * 8 / 1024] * 2, '--', color=colors[i])

        axs[1].plot(thrs, iou_all, '*-', color=colors[i], label=m)
        axs[1].plot([0.1, 0.9], [locals().get(f'baselines_{head}')[0]] * 2, '--k')

        axs[2].plot(thrs, iou_obs, '*-', color=colors[i], label=m)
        axs[2].plot([0.1, 0.9], [locals().get(f'baselines_{head}')[1]] * 2, '--k')

    axs[0].legend(loc='center right')
    axs[0].set_xlabel('Uncertainty threshold')
    axs[0].set_ylabel('CPM size (KB)')
    axs[0].title.set_text('CPM size')

    # axs[1].set_ylim([64, 71])
    axs[1].legend(loc='center right')
    axs[1].set_xlabel('Uncertainty threshold')
    axs[1].set_ylabel('IoU')
    axs[1].title.set_text('IoU all')

    # axs[2].set_ylim([64, 71])
    axs[2].legend(loc='center right')
    axs[2].set_xlabel('Uncertainty threshold')
    axs[2].set_ylabel('IoU')
    axs[2].title.set_text('IoU obs.')

    plt.tight_layout()
    plt.savefig(f"{log_path}/cpm_{head}.png")
    plt.close()


def compare(gevbev_dir, cobevt_dir, gt_dir, out_dir, dataset, mode='conf'):
    gevbev_info_file = f"/mars/projects20/evibev/tmp/gevbev_info_{dataset}.json"
    if not os.path.exists(gevbev_info_file):
        gevbev_info = parse_evibev_inf(os.path.join(gevbev_dir, 'inf'))
        with open(gevbev_info_file, 'w') as fh:
            json.dump(gevbev_info, fh)
    else:
        with open(gevbev_info_file, 'r') as fh:
            gevbev_info = json.load(fh)

    files = sorted(glob.glob(os.path.join(cobevt_dir, '*pth')))
    for f in tqdm.tqdm(files):
        name = os.path.basename(f)[:-4]
        if dataset == 'opv2v':
            name = name.replace('_semantic_lidarcenter', '')
        cobevt_res = torch.load(f)
        gevbev_res = torch.load(gevbev_info[name]['file'])
        cobevt_conf, cobevt_unc = evidence_to_conf_unc(
            cobevt_res['distr_object']['evidence'], False)
        gevbev_conf = gevbev_res['box_bev_conf'][gevbev_info[name]['idx']]

        if dataset == 'opv2v':
            gt_bev = cv2.imread(os.path.join(gt_dir,
                                             cobevt_res['frame_id'][0][0],
                                             cobevt_res['ego_id'][0],
                                             f"{name.split('_')[-1]}_bev_road.png"))
            gt_bev = gt_bev[::-2, ::2]
            gt_obj = np.stack([gt_bev[..., 1]] * 3, axis=-1)
            draw_bev(gt_obj, cobevt_conf, gevbev_conf, out_dir, name, cls_name='obj')

            cobevt_conf, cobevt_unc = evidence_to_conf_unc(
                cobevt_res['distr_surface']['evidence'], False)
            gevbev_conf = gevbev_res['road_confidence'][gevbev_info[name]['idx']]
            gt_sur = np.stack([gt_bev[..., 0]] * 3, axis=-1)
            draw_bev(gt_sur, cobevt_conf, gevbev_conf, out_dir, name, cls_name='sur')
        else:
            gt_bev = cv2.imread(os.path.join(gt_dir, f'{name}.jpg'))
            gevbev_conf = gevbev_conf.transpose(1, 0, 2)
            draw_bev(gt_bev, cobevt_conf, gevbev_conf, out_dir, name, cls_name='obj')


def draw_bev(gt_bev, cobevt_conf, gevbev_conf, out_dir, name, cls_name='obj', mode='conf'):
        h, w = gt_bev.shape[:2]
        cat_img = np.zeros((h * 3 + 6, w, 3))

        if mode == 'cls':
            cobevt_cls = (cobevt_conf.argmax(dim=-1) * 255).squeeze().cpu().numpy()
            gevbev_cls = (gevbev_conf.argmax(dim=-1) * 255).squeeze().cpu().numpy()
            cat_img[:h, :] = render_car(np.sum(gt_bev, axis=-1) > 0)
            cat_img[h + 3:h * 2 + 3, :] = render_car(cobevt_cls)
            cat_img[-h:, :] = render_car(gevbev_cls)
            cv2.imwrite(os.path.join(out_dir, f'{name}_{cls_name}.jpg'), cat_img)
        elif mode == 'conf':
            cat_img[h:h+3] = 255
            cat_img[2*h+3:2*h+6] = 255
            cobevt_conf_img = render_img_hot(cobevt_conf[..., 1].squeeze())
            gevbev_conf_img = render_img_hot(gevbev_conf[..., 1])
            # cv2.imwrite(f.replace('test', 'imgs').replace('pth', 'png'), cobevt_conf_img)
            # cv2.imwrite(os.path.join(os.path.dirname(gevbev_dir), 'imgs', f'{name}.png'), gevbev_conf_img)
            pad = round(cobevt_conf_img.shape[0] - h) // 2
            cat_img[:h, :] = gt_bev
            cat_img[h + 3:h * 2 + 3, :] = cobevt_conf_img[pad:pad+h, pad:pad+w]
            cat_img[-h:, :] = gevbev_conf_img
            cv2.imwrite(os.path.join(out_dir, f'{name}_{cls_name}.jpg'), cat_img)


def load_result(filename):
    iou_all = []
    with open(filename, 'r') as fh:
        for line in fh.readlines():
            if 'iou all' in line:
                iou_all.append(float(line.strip().split(' ')[-1]))
    return iou_all


def pose_err(path, dataset):
    res = {
        'bev-opv2v': {},
        'evibev-opv2v': {},
        'evigausbev-opv2v': {}
    }
    for k in res.keys():
        test_dir = os.path.join(path, dataset)
        files = glob.glob(os.path.join(test_dir, 'test*', 'result.txt'))
        res_with_err = np.zeros((6, 11, 2))
        for f in files:
            test_name = f.split('/')[-2].replace('test', '').split('-')
            if len(test_name) == 1:
                t_std = 0
                r_std = 0
            else:
                t_std = int(test_name[0])
                r_std = int(test_name[1])

            iou_all = load_result(f)
            res_with_err[t_std, r_std, 0] = iou_all[0]
            if len(iou_all) == 2:
                res_with_err[t_std, r_std, 1] = iou_all[1]

        res[k] = {
            'loc': {'road': res_with_err[:, 0, 0], 'object': res_with_err[:, 0, 1]},
            'rot': {'road': res_with_err[0, :, 0], 'object': res_with_err[0, :, 1]},
        }



    # loc err
    xs_loc = np.arange(6) * 0.1
    plt.plot(xs_loc, res_with_err[:, 0, 0])  # road
    plt.plot(xs_loc, res_with_err[:, 0, 1])  # object
    plt.savefig(os.path.join(test_dir, "loc_err.png"))
    plt.close()

    # rot err
    xs_rot = np.arange(11) * 0.1
    plt.plot(xs_rot, res_with_err[0, :, 0])  # road
    plt.plot(xs_rot, res_with_err[0, :, 1])  # object
    plt.savefig(os.path.join(test_dir, "rot_err.png"))
    plt.close()

    with open(os.path.join(test_dir, "pose_err.txt"), 'w') as fh:
        fh.writelines('road loc err:\n')
        fh.writelines('    '.join([f'{x:.1f}' for x in xs_loc]))
        fh.writelines('\n')
        fh.writelines('  '.join([f'{err:02.2f}' for err in res_with_err[:, 0, 0]]))
        fh.writelines('\n')
        fh.writelines('object loc err:\n')
        fh.writelines('    '.join([f'{x:.1f}' for x in xs_loc]))
        fh.writelines('\n')
        fh.writelines('  '.join([f'{err:02.2f}' for err in res_with_err[:, 0, 1]]))
        fh.writelines('\n')
        fh.writelines('road rot err:\n')
        fh.writelines('    '.join([f'{x:.1f}' for x in xs_rot]))
        fh.writelines('\n')
        fh.writelines('  '.join([f'{err:02.2f}' for err in res_with_err[0, :, 0]]))
        fh.writelines('\n')
        fh.writelines('object rot err:\n')
        fh.writelines('    '.join([f'{x:.1f}' for x in xs_rot]))
        fh.writelines('\n')
        fh.writelines('  '.join([f'{err:02.2f}' for err in res_with_err[0, :, 1]]))


def render_img_hot(img):
    if isinstance(img, torch.Tensor):
        img = (img * 255).int().cpu().numpy().astype(np.uint8)
    # out_img = np.zeros_like(img)
    out_img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
    return out_img


def render_car(cls):
    h, w = cls.shape[:2]
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    mask = cls > 0.5
    img[mask, 0] = 128
    img[mask, 1] = 223
    img[mask, 2] = 255
    return img


def draw_cmap_bar():
    img = np.arange(256)
    img = np.stack([img] * 15, axis=0).astype(np.uint8)
    img = render_img_hot(img)
    cv2.imwrite("/mars/publications/gevbev/images/cmap_hot.png", img)

# prc('surface')
# prc('object')
unc_q('surface')
unc_q('object')
# cpm('sur')
# cpm('obj')
# pose_err("/mars/projects20/evibev_exp/v2vreal/evigausbev-opv2v")
# pose_err("/mars/projects20/evibev_exp/v2vreal/evibev-opv2v")
# pose_err("/mars/projects20/evibev_exp/v2vreal/bev-opv2v")
# compare(
#     "/mars/projects20/evibev_exp/v2vreal/evigausbev-opv2v/test0-0",
#     "/mars/projects20/evibev_exp/v2vreal/cobevt/test",
#     "/mars/projects20/evibev_exp/v2vreal/gt_bev",
#     "/mars/projects20/evibev_exp/v2vreal/compare_cobevt_gevbev_conf",
#     mode='conf',
#     dataset='opv2v'
# )
# compare(
#     "/mars/projects20/evibev_exp/opv2v/evigausbev-opv2v/test0-0",
#     "/mars/projects20/evibev_exp/opv2v/cobevt/test",
#     "/koko/OPV2V/additional/test",
#     "/mars/projects20/evibev_exp/opv2v/compare_cobevt_gevbev_conf",
#     mode='conf',
#     dataset='opv2v'
# )
# load_result("/mars/projects20/evibev_exp/opv2v/evigausbev-opv2v/test0-3/result.txt")
# draw_cmap_bar()

