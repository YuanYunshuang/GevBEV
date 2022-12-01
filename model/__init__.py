import importlib, logging
from torch import nn
from model import backbones, submodules


def get_model(cfgs):
    return Model(cfgs)


class Model(nn.Module):
    def __init__(self, cfgs):
        super(Model, self).__init__()
        logging.info("Loading Model ...")
        self.module_order = cfgs['order']
        for module_name in self.module_order:
            setattr(self, module_name,
                self.get_object_instance(module_name, cfgs[module_name])
            )
        modules = [getattr(self, m).__class__.__name__ for m in self.module_order]
        logging.info("Loaded: " + ', '.join(modules))

    def get_object_instance(self, module_name, cfgs):
        try:
            module = importlib.import_module(f'model.backbones.{module_name}')
        except:
            module = importlib.import_module(f'model.submodules.{module_name}')

        cls_name = ''
        for word in  module_name.split('_'):
            cls_name += word[:1].upper() + word[1:]
        cls_obj = getattr(module, cls_name, None)
        assert cls_obj is not None, f'Class \'{cls_name}\' not found.'

        return cls_obj(cfgs)

    def forward(self, batch_dict):
        for module_name in self.module_order:
            getattr(self, module_name)(batch_dict)

    def loss(self, batch_dict):
        loss_total = 0
        loss_dict = {}
        for module_name in self.module_order:
            module = getattr(self, module_name)
            if hasattr(module, 'loss'):
                loss, ldict = module.loss(batch_dict)
                loss_total = loss_total + loss
                loss_dict.update(ldict)
        loss_dict['total'] = loss_total
        return loss_total, loss_dict