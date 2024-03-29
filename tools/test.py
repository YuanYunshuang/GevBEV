import argparse
import os, tqdm, time, glob

from model import get_model
from dataset import get_dataloader
from utils import misc, metrics
from utils.train_utils import *
from config import load_yaml


def test(cfgs, args, n=0):
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

    # load dataset
    test_dataloader = get_dataloader(cfgs['DATASET'], mode='test', use_cuda=args.cuda_loader)

    # check if test result already exists
    outfiles = glob.glob(os.path.join(out_inf_dir, '*.pth'))
    if len(outfiles) == len(test_dataloader):
        i = 0
        if len(outfiles) > 0:
            for f in tqdm.tqdm(outfiles):
                out_dict = torch.load(f)
                i += 1
                for metric in metrics_instances:
                    metric.add_samples(out_dict)
            for metric in metrics_instances:
                metric.summary()
            return

    # load model
    model = get_model(cfgs['MODEL']).to(device)

    # prepare post processor
    test_dataloader.dataset.post_processes.set_vis_dir(out_img_dir)
    post_processor = test_dataloader.dataset.post_process

    # load checkpoint
    # ckpt_file = os.path.join(log_dir, f"epoch{cfgs['TRAIN']['max_epoch']-1}.pth")
    # if not os.path.exists(ckpt_file):
    ckpt_file = os.path.join(log_dir, 'last.pth')
    ckpt = torch.load(ckpt_file)
    logger.write(f"Loaded checkpoint: {ckpt_file}")
    load_model_dict(model, ckpt['model_state_dict'])
    model.eval()

    batch_idx = 0
    with torch.no_grad():
        model.eval()
        for batch_data in tqdm.tqdm(test_dataloader):
            batch_idx += 1
            out_file = os.path.join(out_inf_dir, f"{batch_idx}.pth")
            if os.path.exists(out_file):
                out_dict = torch.load(out_file)
            else:
                load_tensors_to_gpu(batch_data)
                # Forward pass
                model(batch_data)
                # loss_dict = model.loss(batch_data)
                # result.append(loss_dict)
                out_dict = post_processor(batch_data)
                torch.save(out_dict, out_file)

            for metric in metrics_instances:
                metric.add_samples(out_dict)

    for metric in metrics_instances:
        metric.summary()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--cuda_loader", action="store_true")
    parser.add_argument("--save_img", action="store_true")
    args = parser.parse_args()

    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_yaml(args)
    args.save_img = True
    cfgs['DATASET']['postprocessors']['args']['DistributionPostProcess']['visualization'] = args.save_img
    test(cfgs, args)