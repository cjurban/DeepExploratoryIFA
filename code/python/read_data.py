#!/usr/bin/env python
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Some functions for loading data sets.
#
###############################################################################

from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
    
# Read in IPIP FFM data set.
class ipip_ffm_dataset(Dataset):
    def __init__(self, csv_file,
                 which_split,
                 transform = None):
        """
        Args:
            csv_file (string): Path to the CSV file.
            which_split (string): Return the trainng set, the validation set, or the test set.
            transform (Transform): Tranformation of the output samples.
        """
        self.which_split = which_split
        self.transform = transform

        ipip_ffm_data = pd.read_csv(csv_file, sep = ",")
        
        test_size = 0.025
        val_size  = 0.8
        
        if self.which_split == "full":
            self.df = ipip_ffm_data
            
            
        elif self.which_split == "train-only" or self.which_split == "test-only":
            # Split the data into a training set and a test set.
            ipip_ffm_train, ipip_ffm_test = train_test_split(ipip_ffm_data, train_size = 1 - test_size, test_size = test_size, random_state = 45)
            
            if self.which_split == "train-only":
                self.df = ipip_ffm_train
            elif self.which_split == "test-only":
                self.df = ipip_ffm_test
            
        else:
            # Split the data into a training set, a validation set, and a test set.
            ipip_ffm_train, ipip_ffm_test = train_test_split(ipip_ffm_data, train_size = 1 - test_size, test_size = test_size, random_state = 45)
            ipip_ffm_train, ipip_ffm_val = train_test_split(ipip_ffm_train, train_size = 1 - val_size, test_size = val_size, random_state = 50)

            if self.which_split == "train":
                self.df = ipip_ffm_train
            elif self.which_split == "val":
                self.df = ipip_ffm_val
            elif self.which_split == "test":
                self.df = ipip_ffm_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :].to_numpy()

        if self.transform:
            sample = self.transform(sample)

        return sample
    
# Convert a tensor to a data set.
class tensor_dataset(Dataset):
    def __init__(self, tensor):
        """
        Args:
            tensor (Tensor): A Tensor to be converted to a data set.
        """
        self.tensor = tensor

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, idx):
        sample = self.tensor[idx]

        return sample
    
# Convert Numpy arrays in sample to Tensors.
class to_tensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample)