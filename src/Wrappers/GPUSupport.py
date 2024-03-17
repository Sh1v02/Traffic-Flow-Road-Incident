import torch

from src.Utilities.Constants import DEVICE


def optimise(torch_object):
    return torch_object.to(DEVICE)


def tensor(*args, **kwargs):
    return torch.tensor(*args, **kwargs).to(DEVICE)
