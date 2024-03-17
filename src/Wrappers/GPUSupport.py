import torch

from src.Utilities.Constants import DEVICE


def optimise(torch_object):
    return torch_object.to(DEVICE)


def tensor(*args, **kwargs):
    if "dtype" not in kwargs:
        kwargs["dtype"] = torch.float32
    return torch.tensor(*args, **kwargs).to(DEVICE)
