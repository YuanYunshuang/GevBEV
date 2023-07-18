import numpy as np

from test import test
from config import load_yaml
import argparse, os, shutil


def test_loc_err(args, visulize=False):
    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_yaml(args)
    print(args.log_dir)
    exps = np.zeros((16, 2))
    exps[1:6, 0] = np.arange(1, 6) * 0.1
    exps[6:, 1] = np.arange(1, 11) * 0.1
    if args.eval_loc_err:
        for exp in exps:
            if exp.sum() == 0:
                cfgs['DATASET']['loc_err_flag'] = False
            else:
                cfgs['DATASET']['loc_err_flag'] = True
                cfgs['DATASET']['loc_err_t_std'] = exp[0]
                cfgs['DATASET']['loc_err_r_std'] = exp[1]
            cfgs['DATASET']['postprocessors']['args']['DistributionPostProcess'
            ]['visualization'] = visulize
            test(cfgs, args, f"{exp[0] * 10:.0f}-{exp[1] * 10:.0f}")
    else:
        cfgs['DATASET']['loc_err_flag'] = False
        cfgs['DATASET']['postprocessors']['args']['DistributionPostProcess'
        ]['visualization'] = visulize
        test(cfgs, args, "0-0")


def test_cpms(args, visulize=False):
    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_yaml(args)
    cfgs['DATASET']['postprocessors']['args']['DistributionPostProcess'
    ]['visualization'] = visulize
    print(args.log_dir)
    for option in ["road"]:
        cfgs['MODEL']['h_evi_gaus_bev']['args']['cpm_option'] = option
        for n in range(1, 10):
            cfgs['MODEL']['h_evi_gaus_bev']['args']['cpm_thr'] = n * 0.1
            test(cfgs, args, f"_{option}-{n}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/mars/projects20/evibev_exp/opv2v/evigausbev-opv2v")
    parser.add_argument("--eval_loc_err", action="store_true")
    parser.add_argument("--cuda_loader", action="store_true")
    parser.add_argument("--save_img", action="store_true")
    args = parser.parse_args()

    test_cpms(args, args.save_img)
