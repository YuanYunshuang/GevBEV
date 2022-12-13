import matplotlib.pyplot as plt
import numpy as np
import os
import torch

log_dirs = {'evigausbev': 'GEviBEV', 'evibev': 'EviBEV', 'bev': 'BEV'}
log_path = '/media/hdd/yuan/evibev/logs'


def prc(head_name):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    for log_dir, name in log_dirs.items():
        data = torch.load(os.path.join(log_path, f'{log_dir}_p2', 'test',
                                       f'{head_name}_plt_data.pth'))
        pr = data['pr_curve']
        ax.plot(pr[:, 0], pr[:, 1], label=name)
        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 1])
    ax.legend()
    plt.savefig(os.path.join(log_path, f'prc_{head_name}.png'))
    plt.close()


def unc_q(head_name):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()

    thrs = np.arange(0, 1, 0.1)
    colors = ['blue', 'orange', 'green']
    for i, log_dir in enumerate(log_dirs):
        data = torch.load(os.path.join(log_path, f'{log_dir}_p2',
                                       'test', f'{head_name}_plt_data.pth'))
        unc_q = data['unc_Q']
        # ax.plot(thrs + 0.05, unc_q, label=log_dir)
        for thr, r in zip(thrs, unc_q):
            bar = ax.bar(thr + i * 0.03 + 0.02, [r], width=0.03,
                    bottom=[0],
                    color=[colors[i]])
            if thr==0:
                bar.set_label(log_dirs[log_dir])
    ax.plot([0, 1], [1, 0.5], '--k')
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0.5, 1])
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Weighted Accuracy')
    plt.savefig(os.path.join(log_path, f'unc_q_{head_name}.png'))
    plt.close()

# prc('surface')
# prc('object')
unc_q('surface')
unc_q('object')

