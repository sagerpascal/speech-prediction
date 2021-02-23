import torch.nn.functional as F
import torch

def get_loss(conf):
    return torch.nn.NLLLoss()
