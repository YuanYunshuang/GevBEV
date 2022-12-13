from train import train
from config import load_yaml
from utils.misc import setup_logger
import argparse, os, shutil


def try_train(args, exp):
    # try:
    args.log_dir = f"/media/hdd/yuan/evibev/ablation/{exp}"
    # if os.path.exists(os.path.join(args.log_dir, 'test', 'inf')):
    #     shutil.rmtree(os.path.join(args.log_dir, 'test', 'inf'))
    setattr(args, 'config', os.path.join(args.log_dir, 'config.yaml'))
    setattr(args, 'resume', True)
    cfgs = load_yaml(args)
    setup_logger(args.run_name, args.debug)

    train(cfgs)
    # except:
    #     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log_dir", type=str, default="../logs")
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # try_train(args, 'evigausbev_p2')
    try_train(args, 'evibev_p2')
    try_train(args, 'bev_p2')
