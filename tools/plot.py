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
from utils.colors import *


log_dirs = {'evigausbev': 'GevBEV', 'evibev': 'EviBEV', 'bev': 'BEV'}
log_path = '/mars/projects20/evibev_exp'
paths = {
    'bev-opv2v': "/mars/projects20/evibev_exp/bev-opv2v",
    'evibev-opv2v': "/mars/projects20/evibev_exp/evibev-opv2v",
    'evigausbev-opv2v': "/mars/projects20/evibev_exp/evigausbev-opv2v",
    'bev-v2vreal': "/mars/projects20/evibev_exp/bev-v2vreal",
    'evibev-v2vreal': "/mars/projects20/evibev_exp/evibev-v2vreal",
    'evigausbev-v2vreal': "/mars/projects20/evibev_exp/evigausbev-v2vreal",
    'cobevt-opv2v': "/koko/v2vreal-out/cobevt-opv2v",
    'cobevt-v2vreal': "/mars/projects20/evibev_exp/v2vreal/cobevt",
}

cobj1 = [x / 255. for x in [255, 186, 8]]
cobj2 = [x / 255. for x in [220, 47, 2]]
csur1 = [x / 255. for x in [82, 183, 136]]
csur2 = [x / 255. for x in [27, 67, 50]]
# cevibev = [x / 255. for x in [149, 213, 178]]
# cgevbev = [x / 255. for x in [64, 145, 108]]
# ccobevt = [x / 255. for x in [27, 67, 50]]
clinsur = [x / 255. for x in [35, 80, 60]]
clineobj = [x / 255. for x in [55, 6, 23]]


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


def unc_q(head_name, dataset):
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot()

    thrs = np.arange(0, 1, 0.1)
    if head_name == 'surface':
        colors = ['#AAC8A7', '#E9FFC2', '#FDFFAE']
    else:
        colors = ['#FAAB78', '#FFD495', '#FFFBAC']
    for i, log_dir in enumerate(log_dirs):
        data = torch.load(os.path.join(paths[f'{log_dir}-{dataset}'],
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
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0.5, 1])
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Weighted Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(log_path, f'unc_q_{head_name}_{dataset}.png'))
    plt.close()


def cpm(head='obj'):
    log_path = "/mars/projects20/evibev_exp/gevbev-opv2v-cpm"
    hi = 0 if head == 'sur' else 1
    nall = [49685.7, 50585.8]
    nobs = [43466.9, 44344.7]
    nroad = [17463.6, 17916.0]
    baselines_sur = [79.5, 83.1]
    baselines_obj = [74.7, 76.1]
    thrs = np.arange(0.1, 1, 0.1)
    if head == 'obj':
        colors = [cobj1, cobj2]
        cline = clineobj
    else:
        colors = [csur1, csur2]
        cline = clinsur
    fig = plt.figure(figsize=(9, 3))
    axs = fig.subplots(1, 3)
    for i, m in enumerate(['all', 'road']):
        cpm_all = []
        iou_all = []
        iou_obs = []
        for thr in range(1, 10):
            cur_path = f"{log_path}/cpm_{m}/test_{m}-{thr}/result.txt"
            with open(cur_path, 'r') as fh:
                lines = fh.readlines()
                nsur = float(lines[4].strip().split(':')[-1]) * 8 / 1024
                iou_sur_all = float(lines[6].strip().split(' ')[-1])
                iou_sur_obs = float(lines[7].strip().split(' ')[-1])
                nobj = float(lines[12].strip().split(':')[-1]) * 8 / 1024
                iou_obj_all = float(lines[14].strip().split(' ')[-1])
                iou_obj_obs = float(lines[15].strip().split(' ')[-1])
                cpm_all.append(locals().get(f'n{head}'))
                iou_all.append(locals().get(f'iou_{head}_all'))
                iou_obs.append(locals().get(f'iou_{head}_obs'))

        marker = '^-' if m=='road' else 'o-'
        ms = 5
        axs[0].plot(thrs, cpm_all, marker, color=cline, markerfacecolor=colors[i], label=m, markersize=ms)
        axs[0].plot(np.arange(0.1, 1, 0.1), [locals().get(f'n{m}')[hi] * 8 / 1024] * 9, '--', color=colors[i])

        axs[1].plot(thrs, iou_all, marker, color=cline, markerfacecolor=colors[i], label=m, markersize=ms)
        axs[1].plot([0.1, 0.9], [locals().get(f'baselines_{head}')[0]] * 2, '--k')

        axs[2].plot(thrs, iou_obs, marker, color=cline, markerfacecolor=colors[i], label=m, markersize=ms)
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
    iou_all = {}
    k = 'none'
    with open(filename, 'r') as fh:
        for line in fh.readlines():
            if 'surface' in line:
                k = 'road'
            elif 'object' in line:
                k = 'object'
            if 'iou all' in line:
                iou_all[k] =float(line.strip().split(' ')[-1])
    return iou_all


def read_evibev_err(path):
    files = glob.glob(os.path.join(path, 'test*', 'result.txt'))
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
        res_with_err[t_std, r_std, 0] = iou_all.get('road', 0)
        res_with_err[t_std, r_std, 1] = iou_all.get('object', 0)

    res = {
        'loc': {'road': res_with_err[:, 0, 0], 'object': res_with_err[:, 0, 1]},
        'rot': {'road': res_with_err[0, :, 0], 'object': res_with_err[0, :, 1]},
    }
    return res


def read_cobevt_err(filename, has_road=True):
    res_with_err = np.zeros((6, 11, 2))
    with open(filename, 'r') as fh:
        lines = fh.readlines()
        if has_road:
            for l1, l2 in zip(lines[::2], lines[1::2]):
                test_name = os.path.basename(l1.split(':')[0]).replace('test', '').split('-')
                t_std = int(test_name[0])
                r_std = int(test_name[1])
                res_with_err[t_std, r_std, 1] = float(l1.strip().split('[')[1][:-1])
                res_with_err[t_std, r_std, 0] = float(l2.strip().split('[')[1][:-1])
        else:
            for l1 in lines:
                test_name = os.path.basename(l1.split(':')[0]).replace('test', '').split('-')
                t_std = int(test_name[0])
                r_std = int(test_name[1])
                res_with_err[t_std, r_std, 1] = float(l1.strip().split('[')[1][:-1])
    res = {
        'loc': {'road': res_with_err[:, 0, 0], 'object': res_with_err[:, 0, 1]},
        'rot': {'road': res_with_err[0, :, 0], 'object': res_with_err[0, :, 1]},
    }
    return res


def pose_err():
    paths = {
        'bev-opv2v': "/mars/projects20/evibev_exp/bev-opv2v",
        'evibev-opv2v': "/mars/projects20/evibev_exp/evibev-opv2v",
        'evigausbev-opv2v': "/mars/projects20/evibev_exp/evigausbev-opv2v",
        'bev-v2vreal': "/mars/projects20/evibev_exp/bev-v2vreal",
        'evibev-v2vreal': "/mars/projects20/evibev_exp/evibev-v2vreal",
        'evigausbev-v2vreal': "/mars/projects20/evibev_exp/evigausbev-v2vreal",
        'cobevt-opv2v': "/koko/v2vreal-out/cobevt-opv2v",
        'cobevt-v2vreal': "/mars/projects20/evibev_exp/v2vreal/cobevt",
    }
    res_dict = {}
    for k, path in paths.items():
        if 'cobevt' in k:
            res_file = os.path.join(path, 'ious.txt')
            has_road = True if 'opv2v' in k else False
            res = read_cobevt_err(res_file, has_road)
        else:
            res = read_evibev_err(path)

        res_dict[k] = res

    orange = [x / 255. for x in [255, 186, 8]]
    cbev = [x / 255. for x in [82, 183, 136]]
    # cevibev = [x / 255. for x in [149, 213, 178]]
    # cgevbev = [x / 255. for x in [64, 145, 108]]
    # ccobevt = [x / 255. for x in [27, 67, 50]]
    cline1 = [x / 255. for x in [35, 80, 60]]
    cline2 = [x / 255. for x in [55, 6, 23]]

    def plot_err(err_name, bev_name, data_name, ylim):
        if err_name == 'loc':
            xs = np.arange(6) * 0.1
            xlabel = 'Location Error (m)'
            step = 1

        else:
            xs = np.arange(6) * 0.2
            xlabel = 'Rotation Error (°)'
            step=2
        if bev_name == 'road':
            cline = cline1
            cface = cbev
        else:
            cline = cline2
            cface = orange
        plt.figure(figsize=(3, 3))
        plt.plot(xs, res_dict[f'bev-{data_name}'][err_name][bev_name][::step],
                 marker='^', linestyle='--', color=cline, markerfacecolor=cface, label='BEV')
        plt.plot(xs, res_dict[f'evibev-{data_name}'][err_name][bev_name][::step],
                 marker='o', linestyle='--', color=cline, markerfacecolor=cface, label='EviBEV')
        plt.plot(xs, res_dict[f'evigausbev-{data_name}'][err_name][bev_name][::step],
                 marker='*', linestyle='--', color=cline, markerfacecolor=cface, label='GevBEV')
        plt.plot(xs, res_dict[f'cobevt-{data_name}'][err_name][bev_name][::step],
                 marker='s', linestyle='--', color='k', markerfacecolor='k', label='CoBEVT')
        plt.legend(fontsize=8)
        plt.xlabel(xlabel)
        plt.ylim(ylim)
        plt.ylabel('IoU (%)')
        plt.tight_layout()
        plt.savefig(f"/home/yuan/Pictures/pose_error/{data_name}_{err_name}_err_{bev_name}.png")
        plt.close()
    # opv2v loc err, road
    plot_err('loc', 'road', 'opv2v', [65, 81])
    plot_err('rot', 'road', 'opv2v', [65, 81])
    plot_err('loc', 'object', 'opv2v', [50, 80])
    plot_err('rot', 'object', 'opv2v', [50, 80])
    plot_err('loc', 'object', 'v2vreal', [25, 50])
    plot_err('rot', 'object', 'v2vreal', [25, 50])


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


def draw_custom_cmap_bar(colors):
    np.array(colors).reshape(-1, 3)
    img = np.stack([colors] * 15, axis=0).astype(np.uint8)
    img = cv2.resize(img, (256, 15), interpolation=cv2.INTER_LINEAR)
    return img


def draw_road_object_bev(data_dict, data_dir, road_color_palette, object_color_palette, model):
    if 'road_confidence' in data_dict:
        road_confs = data_dict['road_confidence']
        object_confs = data_dict['box_bev_conf']
    else:
        road_confs = data_dict['distr_surface']['evidence'][:, 3:253, 3:253]
        object_confs = data_dict['distr_object']['evidence'][:, 3:253, 3:253]
    imgs_dict = {}

    def draw_bev(road_conf, object_conf):
        road_colors_idx = (road_conf * 255).astype(np.uint8)
        road_colors = road_color_palette[road_colors_idx.reshape(-1)].reshape(250, 250, 3)
        object_colors_idx = (object_conf * 255).astype(np.uint8)
        object_colors = object_color_palette[object_colors_idx.reshape(-1)].reshape(250, 250, 3)
        object_mask = object_conf > 0.5
        road_colors[object_mask] = object_colors[object_mask]
        return road_colors

    for i, frame_id in enumerate(data_dict['frame_id']):
        name = f"{frame_id[0]}_{frame_id[1]}"
        # get ego_id
        if 'ego_id' in data_dict:
            ego_id = data_dict['ego_id'][i]
        else:
            scenario_path = os.path.join(data_dir, frame_id[0])
            cav_list = sorted([x for x in os.listdir(scenario_path)
                               if os.path.isdir(os.path.join(scenario_path, x))])
            ego_id = cav_list[0]
        gt_bev = cv2.imread(os.path.join(data_dir, frame_id[0],
                                         ego_id, frame_id[1].replace('_semantic_lidarcenter', '') + '_bev.png'))[::2, ::2]
        cur_road_conf = road_confs[i, :, :, 1].cpu().numpy()
        cur_object_conf = object_confs[i, :, :, 1].cpu().numpy()
        gt_road = gt_bev[..., 0] / 255
        gt_object = gt_bev[..., 1] / 150
        gt_object[gt_object < 1] = 0
        imgs_dict[name] = {
            f'pred_{model}': draw_bev(cur_road_conf, cur_object_conf),
            f'gt_{model}': draw_bev(gt_road, gt_object),
        }

    return imgs_dict


def bev_img_for_pub(inf_dir, data_dir, model):
    road_color_palette = draw_custom_cmap_bar(summer_ocean)[0]
    object_color_palette = draw_custom_cmap_bar(summer_desert)[0]
    files = sorted(glob.glob(os.path.join(inf_dir, '*.pth')))
    for f in files:
        data_dict = torch.load(f)
        imgs_dict = draw_road_object_bev(
            data_dict, data_dir, road_color_palette, object_color_palette, model
        )
        for k, v in imgs_dict.items():
            for x, img in v.items():
                cv2.imwrite(f"/home/yuan/Pictures/{x}/{k}.png", img[..., ::-1])


def stech_imgs(path):
    files = glob.glob(os.path.join(path, 'pred/*.png'))
    for f in files:
        img_gevbev = cv2.imread(f) / 255.
        img_cobevt = cv2.imread(f"/home/yuan/Pictures/pred_cobevt/{os.path.basename(f)[:-4]}_semantic_lidarcenter.png") / 255.
        img_gt = cv2.imread(f"/home/yuan/Pictures/gt/{os.path.basename(f)}")[::-1] / 255.

        fig = plt.figure(figsize=(12, 5))
        axs = fig.subplots(1, 3)
        axs[0].imshow(img_gt[..., ::-1])
        axs[0].set_title('Ground Truth')
        axs[0].axis('off')
        axs[1].imshow(img_cobevt[..., ::-1])
        axs[1].set_title('CoBEVT')
        axs[1].axis('off')
        axs[2].imshow(img_gevbev[..., ::-1])
        axs[2].set_title('GevBEV')
        axs[2].axis('off')
        plt.tight_layout()
        plt.savefig(f.replace('pred', 'steched'))
        plt.close()


# prc('surface')
# prc('object')
# unc_q('surface', 'opv2v')
# unc_q('object', 'opv2v')
# unc_q('object', 'v2vreal')
cpm('sur')
cpm('obj')
# pose_err()
# compare(
#     "/mars/projects20/evibev_exp/v2vreal/evigausbev-opv2v/test0-0",
#     "/mars/projects20/evibev_exp/v2vreal/cobevt/test",
#     "/mars/projects20/evibev_exp/v2vreal/gt_bev",
#     "/mars/projects20/evibev_exp/v2vreal/compare_cobevt_gevbev_conf",
#     mode='conf',
#     dataset='opv2v'
# )
# compare(
#     "/mars/projects20/evibev_exp/opv2v/evigausbev/test0-0",
#     "/koko/v2vreal-out/cobevt-opv2v/test0-0",
#     "/koko/OPV2V/additional/test",
#     "/mars/projects20/evibev_exp/opv2v/compare_cobevt_gevbev_conf",
#     mode='conf',
#     dataset='opv2v'
# )
# load_result("/mars/projects20/evibev_exp/opv2v/evigausbev-opv2v/test0-3/result.txt")
# draw_cmap_bar()
# bev_img_for_pub("/koko/v2vreal-out/cobevt-opv2v/test0-0",
#                 "/koko/OPV2V/augmented/test", 'cobevt')
# draw_custom_cmap_bar(summer_ocean)
# stech_imgs("/home/yuan/Pictures/scene2")
