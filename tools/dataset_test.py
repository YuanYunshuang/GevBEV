import torch
import hydra
from pathlib import Path
from dataset.nuscenes_dataset import NuscenesDataset


@hydra.main(config_path=str(Path(__file__).parents[1] / 'config'),
            config_name='minkunet_evigausbev_nuscenes.yaml')
def main(cfg):
    dataset = NuscenesDataset(cfg['DATASET'], mode='train')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=8,
                                             sampler=None, num_workers=4,
                                             shuffle=False,
                                             collate_fn=dataset.collate_batch)
    for batch in dataloader:
        print(batch.keys())


if __name__ == '__main__':
    main()