import os
import glob
import tqdm
import argparse

import cv2
from utils import misc, metrics
from utils.train_utils import *
from config import load_yaml
import dataset.processors as PP


def eval_v2vreal_official(args, obsmasks=None):
    cfgs = load_yaml(args)

    log_dir_in = args.test_dir
    post_processor = []
    for name in cfgs['postprocessors']['test']:
        post_processor.append(getattr(PP, name)(**cfgs['postprocessors']['args'][name]))
    post_processor = PP.Compose(post_processor)

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


def iou(pred, gt, obsmsk=None):
    if obsmsk is not None:
        pred = pred[obsmsk]
        gt = gt[obsmsk]
    intersection = torch.logical_and(pred, gt)
    union = torch.logical_or(pred, gt)
    iou_bev = intersection.sum() / union.sum()
    return iou_bev


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="/mars/projects20/evibev_exp/opv2v/attnfuse/test")
    parser.add_argument("--data_test_dir", type=str, default="/koko/OPV2V/additional/test")
    parser.add_argument("--config", type=str, default="../config/opv2v_test.yaml")
    args = parser.parse_args()

    eval_v2vreal_official(args)

