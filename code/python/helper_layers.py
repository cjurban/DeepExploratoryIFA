#!/usr/bin/env python
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Helpful layers for building models.
#
###############################################################################

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import *

# Module for a bias-only layer.
class Bias(nn.Module):
    
    def __init__(self,
                 init_vals):
        """
        Args:
            init_vals (Tensor): Tensor of floats containing initial values.
        """
        super(Bias, self).__init__()
        self.bias = nn.Parameter(init_vals)

    def forward(self, x):
        return self.bias + x