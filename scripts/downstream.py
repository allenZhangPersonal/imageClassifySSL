import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from PIL import Image # PIL is a library to process images

import constants
