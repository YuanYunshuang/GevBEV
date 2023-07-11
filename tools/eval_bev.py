import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import argparse

import cv2
import torch.nn.functional as F

from model import get_model
from dataset import get_dataloader
from utils import misc, metrics
from utils.train_utils import *
from config import load_yaml
import dataset.processors as PP


def eval_v2vreal_official(args, obsmasks=None):
    cfgs = load_yaml(args)

    # set paths
    log_dir_in = args.test_dir
    # log_dir_out = os.path.join(os.path.dirname(log_dir_in), 'test_evibev')
    # out_img_dir = os.path.join(log_dir_out, f'test{n}', 'img')
    # misc.ensure_dir(out_img_dir)
    # logger = open(os.path.join(log_dir_out, f'test{n}', 'result.txt'), mode='w')

    post_processor = []
    for name in cfgs['postprocessors']['test']:
        post_processor.append(getattr(PP, name)(**cfgs['postprocessors']['args'][name]))
    post_processor = PP.Compose(post_processor)
    # post_processor.set_vis_dir(log_dir_in)

    metrics_instances = []
    for metric_name in cfgs['metrics']:
        metric_cls = getattr(metrics, metric_name, None)
        if metric_cls is not None:
            metrics_instances.append(metric_cls(cfgs['metrics'][metric_name],
                                                log_dir_in, None, name=f'test_tmp'))

    # get bev-opv2v map size
    bev_sizes = {}
    pp_args = cfgs['postprocessors']['args']['DistributionPostProcess']
    lr = pp_args['lidar_range']
    vs = pp_args['voxel_size']
    for k, v in pp_args['stride'].items():
        bev_sizes[k] = {
            'x': round((lr[3] - lr[0]) / vs[0] / v),
            'y': round((lr[4] - lr[1]) / vs[1] / v)
        }

    outfiles = sorted(glob.glob(os.path.join(log_dir_in, '*.pth')))

    ious_bev_dynamic = []
    ious_bev_static = []
    i = 0
    for f in tqdm.tqdm(outfiles):
        # i += 1
        # if i >= 198:
        #     print('debug')
        out_dict = torch.load(f)

        bevmap_static = None
        bevmap_dynamic= None
        obsmsk = None

        if 'surface' not in bev_sizes:
            out_dict.pop('distr_surface')
        else:
            scenario = out_dict['frame_id'][0][0]
            frame = out_dict['frame_id'][0][1].split('_')[0]

            # get observation mask
            if obsmasks is not None:
                obsmsk = obsmasks['_'.join([scenario, frame])]

            # load gt bev-opv2v map
            bev_file = os.path.join(
                args.data_test_dir,
                scenario,
                out_dict['ego_id'][0],
                frame + '_bev_road.png',
            )
            bevmap = cv2.imread(bev_file)[::-2, ::2, :2].copy()
            bevmap_static = bevmap[..., :1].transpose(2, 0, 1)
            bevmap_static = torch.from_numpy(bevmap_static).cuda()
            bevmap_dynamic = bevmap[..., 1:].transpose(2, 0, 1)
            bevmap_dynamic = torch.from_numpy(bevmap_dynamic).cuda()
            out_dict['bevmap_static'] = bevmap_static
            out_dict['bevmap_dynamic'] = bevmap_dynamic

        # out_dict = format_to_evibev(out_dict)
        out_dict = post_processor(out_dict)
        for metric in metrics_instances:
            metric.remove_ego_box(out_dict)
            # cal object iou
            # if bevmap_dynamic is None:
            bevmap_dynamic = metric.get_gt_mask(out_dict, len(out_dict['box_bev_conf']))[0].float()
            bev_pred_dynamic = (out_dict['box_bev_conf'][0].argmax(dim=-1) == 1).float()
            # frameworks in the repo v2v4real can only process bev-opv2v shape that is dividable by 2**n, n > 8
            # for opv2v dataset we train with shape (256, 256) but evaluate on (250, 250) to match the size
            # of cobevt (camera track). Therefore, we crop here the bev-opv2v maps to the goal size.
            sz = bev_sizes['object']
            px = (bev_pred_dynamic.shape[0] - sz['x']) // 2
            py = (bev_pred_dynamic.shape[1] - sz['y']) // 2
            bev_pred_dynamic = bev_pred_dynamic[px:sz['x']+px, py:sz['y']+py]
            ious_bev_dynamic.append(iou(bev_pred_dynamic, bevmap_dynamic.squeeze(), obsmsk))

            # cal. surface iou
            if bevmap_static is not None:
                bev_pred_static = (out_dict['road_confidence'][0].argmax(dim=-1) == 1).float()
                sz = bev_sizes['surface']
                px = (bev_pred_static.shape[0] - sz['x']) // 2
                py = (bev_pred_static.shape[1] - sz['y']) // 2
                bev_pred_static = bev_pred_static[px:sz['x'] + px, py:sz['y'] + py]
                ious_bev_static.append(iou(bev_pred_static, bevmap_static.squeeze(), obsmsk))

    with open((os.path.join(os.path.dirname(log_dir_in), 'ious.txt')), 'a') as fh:
        iou_bev_dynamic_all = torch.stack(ious_bev_dynamic).mean().item() * 100
        print('bev-opv2v dynamic', iou_bev_dynamic_all)
        fh.write(f'{args.test_dir}: dynamic [{iou_bev_dynamic_all:.2f}]\n')
        if len(ious_bev_static) > 0:
            iou_bev_static_all = torch.stack(ious_bev_static).mean().item() * 100
            print('bev-opv2v static', iou_bev_static_all)
            fh.write(f'{" " * len(args.test_dir)}: static  [{iou_bev_static_all:.2f}]\n')


def eval_evibev(args):
    cfgs = load_yaml(args)

    # load info
    evibev_info = parse_evibev_inf(args.evibev_dir)

    # set paths
    log_dir_in = args.test_dir
    log_dir_out = os.path.join(os.path.dirname(log_dir_in), 'test_evibev')
    out_img_dir = os.path.join(log_dir_out, f'test_tmp', 'img')
    misc.ensure_dir(out_img_dir)
    logger = open(os.path.join(log_dir_out, f'test_tmp', 'result.txt'), mode='w')

    metrics_instances = []
    for metric_name in cfgs['metrics']:
        metric_cls = getattr(metrics, metric_name, None)
        if metric_cls is not None:
            metrics_instances.append(metric_cls(cfgs['metrics'][metric_name],
                                                log_dir_out, logger, name=f'test_tmp'))
    ious_evibev = []
    for name, info in tqdm.tqdm(evibev_info.items()):
        evibev_data = torch.load(info['file'])
        for metric in metrics_instances:
            bev_gt = metric.get_gt_mask(evibev_data, len(evibev_data['box_bev_conf']))[info['idx']].float()
            # evibev_pred = evibev_data['box_bev_conf'][info['idx'], :, :, 1]
            evibev_pred = (evibev_data['box_bev_conf'][info['idx']].argmax(dim=-1) == 1).float()
            iou_evibev = iou(evibev_pred, bev_gt)
            ious_evibev.append(iou_evibev)

    iou_evibev_all = torch.stack(ious_evibev).mean().item() * 100
    print('evibev-opv2v', iou_evibev_all)
    open(f'iou_{iou_evibev_all:.2f}.txt').close()


def format_to_evibev(out_dict):
    out_dict['distr_object']['evidence'] = \
        out_dict['distr_object']['evidence'].permute(0, 2, 1, 3)
    out_dict['detection']['pred_boxes'] = \
        F.pad(out_dict['detection']['pred_boxes'], (1, 0), mode='constant', value=0)
    out_dict['target_boxes'] = \
        F.pad(out_dict['target_boxes'], (1, 0), mode='constant', value=0)
    return out_dict


def load_obs_mask(obsmask_dir):
    files = glob.glob(os.path.join(obsmask_dir, '*.pth'))
    obsmask = {}
    for f in tqdm.tqdm(files):
        data_dict = torch.load(f)
        for i, frame_id in enumerate(data_dict['frame_id']):
            name = f"{frame_id[0]}_{frame_id[1]}"
            obsmask[name] = data_dict['box_obs_mask'][i]

    torch.save(obsmask, "../tmp/obsmask_evibev.pth")
    return obsmask


def parse_evibev_inf(inf_dir):
    files = glob.glob(os.path.join(inf_dir, '*.pth'))
    res = {}
    for f in files:
        data_dict = torch.load(f)
        for i, frame_id in enumerate(data_dict['frame_id']):
            name = f"{frame_id[0]}_{frame_id[1]}"
            res[name] = {'file': f, 'idx': i}

    return res


def iou(pred, gt, obsmsk=None):
    if obsmsk is not None:
        pred = pred[obsmsk]
        gt = gt[obsmsk]
    intersection = torch.logical_and(pred, gt)
    union = torch.logical_or(pred, gt)
    iou_bev = intersection.sum() / union.sum()
    return iou_bev


def plot_result(bev_pred, evibev_pred, bev_gt):
    bev_pred = bev_pred.cpu().numpy().T
    evibev_pred = evibev_pred.cpu().numpy().T
    bev_gt = bev_gt.cpu().numpy().T
    h, w = bev_gt.shape
    fig = plt.figure(figsize=(h / 50 * 3, w / 50))
    # fig.suptitle(f'IoU: bev-opv2v({iou_bev * 100:.2f}), evibev-opv2v({iou_evibev * 100:.2f})')
    axs = fig.subplots(3, 1)
    axs[0].imshow(bev_pred)
    axs[1].imshow(evibev_pred)
    axs[2].imshow(bev_gt)
    plt.savefig("/home/yuan/Downloads/tmp.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="/mars/projects20/evibev_exp/opv2v/attnfuse/test")
    # parser.add_argument("--data_test_dir", type=str, default="/koko/v2vreal/test")
    # parser.add_argument("--config", type=str, default="../config/v2vreal_test.yaml")
    parser.add_argument("--data_test_dir", type=str, default="/koko/OPV2V/additional/test")
    parser.add_argument("--config", type=str, default="../config/opv2v_test.yaml")
    args = parser.parse_args()

    # eval_v2vreal_official(args)

    test_dirs = sorted(glob.glob("/mars/projects20/evibev_exp/opv2v/cobevt/test*-*"))
    for test_dir in test_dirs:
        # if test_dir == "/mars/projects20/evibev_exp/v2vreal/cobevt/test0-1":
        #     continue
        print(test_dir)
        args.test_dir = test_dir
        eval_v2vreal_official(args)
