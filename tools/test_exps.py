from test import test
from config import load_yaml
import argparse, os, shutil


def try_test(args, exp, n=0):
    # try:
    args.log_dir = f"/media/hdd/yuan/evibev_exp/ablation/{exp}"
    # if os.path.exists(os.path.join(args.log_dir, 'test', 'inf')):
    #     shutil.rmtree(os.path.join(args.log_dir, 'test', 'inf'))
    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_yaml(args)
    cfgs['DATASET']['postprocessors']['args']['DistributionPostProcess']['vis_dir'] = \
        f'/media/hdd/yuan/evibev_exp/ablation/evigausbev_cpm/test{n}/img'
    cfgs['MODEL']['h_evi_gaus_bev']['args']['cpm_thr'] = n * 0.1
    test(cfgs, args, n)
    # except:
    #     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--vis-func", type=str) # , default="vis_semantic_unc"
    args = parser.parse_args()

    for n in range(5, 10):
        try_test(args, 'evigausbev_cpm', n)
    # try_test(args, 'evibev_p2')
    # try_test(args, 'bev_p2')
