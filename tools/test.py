import argparse
import os, tqdm, time, glob

import torch.multiprocessing as mp

from model import get_model
from dataset import get_dataloader
from utils import misc, metrics
from utils.train_utils import *
from config import load_yaml


def test(cfgs, args, n=0):
    mp.set_start_method('spawn')
    seed_everything(1234)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load checkpoint
    log_dir = cfgs['TRAIN']['log_dir']

    # set paths
    out_img_dir = os.path.join(log_dir, f'test{n}', 'img')
    out_inf_dir = os.path.join(log_dir, f'test{n}', 'inf')
    misc.ensure_dir(out_img_dir)
    misc.ensure_dir(out_inf_dir)
    logger = open(os.path.join(log_dir, f'test{n}', 'result.txt'), mode='w')

    metrics_instances = []
    for metric_name in cfgs['TEST']['metrics']:
        metric_cls = getattr(metrics, metric_name, None)
        if metric_cls is not None:
            metrics_instances.append(metric_cls(cfgs['TEST']['metrics'][metric_name],
                                                log_dir, logger, name=f'test{n}'))

    # outfiles = glob.glob(os.path.join(out_inf_dir, '*.pth'))
    # i = 0
    # if len(outfiles) > 0:
    #     for f in tqdm.tqdm(outfiles):
    #         # if i > 20:
    #         #     break
    #         out_dict = torch.load(f)
    #         i += 1
    #         for metric in metrics_instances:
    #             metric.add_samples(out_dict)
    #     for metric in metrics_instances:
    #         metric.summary()
    #     return

    # load models
    test_dataloader = get_dataloader(cfgs['DATASET'], mode='test',)
    model = get_model(cfgs['MODEL']).to(device)

    # prepare post processor
    test_dataloader.dataset.post_processes.set_vis_dir(out_img_dir)
    post_processor = test_dataloader.dataset.post_process

    # load checkpoint
    ckpt_file = os.path.join(log_dir, f"epoch{cfgs['TRAIN']['max_epoch']-1}.pth")
    if not os.path.exists(ckpt_file):
        ckpt_file = os.path.join(log_dir, 'last.pth')
    ckpt = torch.load(ckpt_file)
    logger.write(f"Loaded checkpoint: {ckpt_file}")
    load_model_dict(model, ckpt['model_state_dict'])
    model.eval()



    result = []
    batch_idx = 0
    with torch.no_grad():
        model.eval()
        for batch_data in tqdm.tqdm(test_dataloader):
            batch_idx += 1
            # if batch_idx > 3:
            #     break
            load_tensors_to_gpu(batch_data)
            # Forward pass
            model(batch_data)
            # loss_dict = model.loss(batch_data)
            # result.append(loss_dict)
            out_dict = post_processor(batch_data)
            # torch.save(out_dict, os.path.join(log_dir, f'test{n}', 'inf', f"{batch_idx}.pth"))
            for metric in metrics_instances:
                metric.add_samples(out_dict)

    for metric in metrics_instances:
        metric.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, default="../logs")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--vis-func", type=str) # , default="vis_semantic_unc"
    args = parser.parse_args()

    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_yaml(args)

    test(cfgs, args)