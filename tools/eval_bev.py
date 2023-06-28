import os
import glob
import tqdm
import argparse

import torch
import torch.nn.functional as F

from model import get_model
from dataset import get_dataloader
from utils import misc, metrics
from utils.train_utils import *
from config import load_yaml
import dataset.processors as PP


def eval(args, n=''):
    cfgs = load_yaml(args)

    # load checkpoint
    log_dir_in = args.test_dir
    log_dir_out = os.path.join(os.path.dirname(log_dir_in), 'test_evibev')

    # set paths
    out_img_dir = os.path.join(log_dir_out, f'test{n}', 'img')
    misc.ensure_dir(out_img_dir)
    logger = open(os.path.join(log_dir_out, f'test{n}', 'result.txt'), mode='w')

    post_processor = []
    for name in cfgs['postprocessors']['test']:
        post_processor.append(getattr(PP, name)(**cfgs['postprocessors']['args'][name]))
    post_processor = PP.Compose(post_processor)
    post_processor.set_vis_dir(out_img_dir)

    metrics_instances = []
    for metric_name in cfgs['metrics']:
        metric_cls = getattr(metrics, metric_name, None)
        if metric_cls is not None:
            metrics_instances.append(metric_cls(cfgs['metrics'][metric_name],
                                                log_dir_out, logger, name=f'test{n}'))

    outfiles = sorted(glob.glob(os.path.join(log_dir_in, '*.pth')))

    for f in tqdm.tqdm(outfiles):
        out_dict = torch.load(f)
        out_dict = format_to_evibev(out_dict)
        out_dict = post_processor(out_dict)
        for metric in metrics_instances:
            metric.add_samples(out_dict)
    for metric in metrics_instances:
        metric.summary()


def format_to_evibev(out_dict):
    out_dict['distr_object']['evidence'] = \
        out_dict['distr_object']['evidence'].permute(0, 2, 1, 3)
    out_dict['detection']['pred_boxes'] = \
        F.pad(out_dict['detection']['pred_boxes'], (1, 0), mode='constant', value=0)
    out_dict['target_boxes'] = \
        F.pad(out_dict['target_boxes'], (1, 0), mode='constant', value=0)
    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="/mars/projects20/v2vreal-out/corpbevtlidar/test")
    parser.add_argument("--config", type=str, default="./config/baseline_test.yaml")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--vis-func", type=str) # , default="vis_semantic_unc"
    args = parser.parse_args()

    eval(args)