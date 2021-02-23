from torch import optim
import torch


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def get_optimizer(conf, model):
    if conf['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=conf['lr'], weight_decay=conf['weight_decay'])
    else:
        raise AttributeError("Unsupported optimizer in config file: {}".format(conf['optimizer']))

