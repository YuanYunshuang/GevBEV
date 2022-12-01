import re
import yaml
import os

import numpy as np


def load_yaml(args):
    """
    Load yaml config file and return a dictionary.
    :param args: configs
    :return: parameter dictionary
    """

    with open(args.config, 'r') as stream:
        loader = yaml.Loader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        params = yaml.load(stream, Loader=loader)
        # update params
        if hasattr(args, 'run_name'):
            params['TRAIN']['run_name'] = args.run_name
        if hasattr(args, 'log_dir'):
            params['TRAIN']['log_dir'] = args.log_dir
        if hasattr(args, 'resume'):
            params['TRAIN']['resume'] = args.resume

    return params


def save_yaml(data, save_path):
    """
    Save data to yaml file
    :param data: dict, hyperparameters
    :param filename: str,
    """
    data['TRAIN']['save_path'] = save_path
    filename = os.path.join(save_path, "config.yaml")
    with open(filename, 'w') as fid:
        yaml.dump(data, fid, default_flow_style=False)