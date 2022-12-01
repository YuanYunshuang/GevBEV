from torch import nn
from model.submodules.utils import minkconv_conv_block


class MCompression(nn.Module):
    def __init__(self, cfgs):
        super(MCompression, self).__init__()

        self.strides = []
        for cfg in cfgs:
            k = list(cfg.keys())[0]
            v = cfg[k]
            self.strides.append(int(k[1]))
            layers = []
            steps = v['steps']
            channels = v['channels']
            for i, s in enumerate(steps):
                step = [1, 1, s]
                layers.append(
                    minkconv_conv_block(channels[i], channels[i+1],
                                        step, step, 3, 0.1)
                )
            layers = nn.Sequential(*layers)
            setattr(self, f'{k}_compression', layers)

    def forward(self, batch_dict):
        stensors = batch_dict['backbone']
        out = {}

        for stride in self.strides:
            out[f'p{stride}'] = \
                getattr(self, f'p{stride}_compression')(stensors[f'p{stride}'])
        batch_dict['compression'] = out