from __future__ import print_function
import os
import logging
import numpy
import subprocess
import time
from datetime import timedelta
import torch

def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm(params, clip_th)
    return (not numpy.isfinite(befgad) or (befgad > ignore_th))

def load_to_cpu(path):
    model = torch.load(path, map_location=lambda storage, loc: storage)
    model.cpu()
    return model
