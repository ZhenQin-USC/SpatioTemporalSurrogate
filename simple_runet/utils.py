import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from os.path import join
from collections import OrderedDict
import h5py
from torch.utils.data import Dataset #, DataLoader
from tqdm import tqdm
import os
import json
from torch.utils.data import Dataset, DataLoader
import psutil


def memory_usage_psutil():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss/(1e3)**3
    print('Memory Usage in Gb: {:.2f}'.format(mem))  # in GB 
    return mem
