import matplotlib.pyplot as plt
import numpy as np
import os
import torch

log_dirs = {'evigausbev': 'GEviBEV', 'evibev': 'EviBEV', 'bev': 'BEV'}
log_path = '/media/hdd/yuan/evibev_exp/ablation'


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


def cpm(head='obj'):
    hi = 0 if head == 'sur' else 1
    nall = [45783.4, 46698.0]
    nobs = [43466.9, 44344.7]
    nroad = [16271.6, 16748.1]
    baselines_sur = [67.4, 70.3]
    baselines_obj = [74.5, 74.8]
    thrs = np.arange(0.1, 1, 0.1)
    colors = ['blue', 'green']
    fig = plt.figure(figsize=(9, 3))
    axs = fig.subplots(1, 3)
    for i, m in enumerate(['all', 'road']):
        cpm_all = []
        iou_all = []
        iou_obs = []
        for thr in range(1, 10):
            cur_path = f"{log_path}/evigausbev_cpm_{m}/test{thr}/result.txt"
            with open(cur_path, 'r') as fh:
                lines = fh.readlines()
                nsur = float(lines[2].strip().split(':')[-1]) * 8 / 1024
                iou_sur_all = float(lines[4].strip().split(' ')[-1])
                iou_sur_obs = float(lines[5].strip().split(' ')[-1])
                nobj = float(lines[8].strip().split(':')[-1]) * 8 / 1024
                iou_obj_all = float(lines[10].strip().split(' ')[-1])
                iou_obj_obs = float(lines[11].strip().split(' ')[-1])
                cpm_all.append(locals().get(f'n{head}'))
                iou_all.append(locals().get(f'iou_{head}_all'))
                iou_obs.append(locals().get(f'iou_{head}_obs'))

        axs[0].plot(thrs, cpm_all, '*-', color=colors[i], label=m)
        axs[0].plot([0.1, 0.9], [locals().get(f'n{m}')[hi] * 8 / 1024] * 2, '--', color=colors[i])

        axs[1].plot(thrs, iou_all, '*-', color=colors[i], label=m)
        axs[1].plot([0.1, 0.9], [locals().get(f'baselines_{head}')[0]] * 2, '--k')

        axs[2].plot(thrs, iou_obs, '*-', color=colors[i], label=m)
        axs[2].plot([0.1, 0.9], [locals().get(f'baselines_{head}')[1]] * 2, '--k')

    axs[0].legend(loc='center right')
    axs[0].set_xlabel('Uncertainty threshold')
    axs[0].set_ylabel('CPM size (KB)')
    axs[0].title.set_text('CPM size')

    # axs[1].set_ylim([64, 71])
    axs[1].legend(loc='center right')
    axs[1].set_xlabel('Uncertainty threshold')
    axs[1].set_ylabel('IoU')
    axs[1].title.set_text('IoU all')

    # axs[2].set_ylim([64, 71])
    axs[2].legend(loc='center right')
    axs[2].set_xlabel('Uncertainty threshold')
    axs[2].set_ylabel('IoU')
    axs[2].title.set_text('IoU obs.')

    plt.tight_layout()
    plt.savefig(f"{log_path}/cpm_{head}.png")
    plt.close()



# prc('surface')
# prc('object')
# unc_q('surface')
# unc_q('object')
cpm('sur')
cpm('obj')

