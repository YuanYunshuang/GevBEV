from test import test
from config import load_yaml
import argparse, os, shutil


def try_test(args, exp):
    # try:
    args.log_dir = f"/media/hdd/yuan/evibev/logs/{exp}"
    # if os.path.exists(os.path.join(args.log_dir, 'test', 'inf')):
    #     shutil.rmtree(os.path.join(args.log_dir, 'test', 'inf'))
    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    cfgs = load_yaml(args)
    test(cfgs, args)
    # except:
    #     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--save-img", action="store_true")
    parser.add_argument("--vis-func", type=str) # , default="vis_semantic_unc"
    args = parser.parse_args()

    try_test(args, 'evigausbev')
    try_test(args, 'evibev')
    try_test(args, 'bev')
