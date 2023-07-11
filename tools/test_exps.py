import numpy as np

from test import test
from config import load_yaml
import argparse, os, shutil


def try_test(args, exp, visulize=False):
    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_yaml(args)
    if exp.sum() == 0:
        cfgs['DATASET']['loc_err_flag'] = False
    else:
        cfgs['DATASET']['loc_err_flag'] = True
        cfgs['DATASET']['loc_err_t_std'] = exp[0]
        cfgs['DATASET']['loc_err_r_std'] = exp[1]
    cfgs['DATASET']['postprocessors']['args']['DistributionPostProcess']['visualization'] = visulize
    test(cfgs, args, f"{exp[0] * 10:.0f}-{exp[1] * 10:.0f}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/mars/projects20/evibev_exp/opv2v/evigausbev-opv2v")
    parser.add_argument("--eval_loc_err", action="store_true")
    parser.add_argument("--cuda_loader", action="store_true")
    parser.add_argument("--save_img", action="store_true")
    args = parser.parse_args()

    print(args.log_dir)
    exps = np.zeros((16, 2))
    exps[1:6, 0] = np.arange(1, 6) * 0.1
    exps[6:, 1] = np.arange(1, 11) * 0.1

    if args.eval_loc_err:
        for exp in exps[8:]:
            print(exp)
            try_test(args, exp, args.save_img)
    else:
        try_test(args, exps[0], args.save_img)
