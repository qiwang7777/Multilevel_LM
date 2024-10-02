#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:06:35 2024

@author: wang
"""

import torch


from Multilevel_LM.mlm_main.MLM_TR import MLM_TR
import numpy as np
import matplotlib.pyplot as plt

def test_func_1d(x):
    return torch.sin(x)
from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
input_dim = 1
output_dim = 1
n_hidden_layers = 1
r_nodes_per_layer = 300
model_21 = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim)
sample_num = 41
x_1d = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,1), dtype=torch.float32)
pred_1d = MLM_TR(test_func_1d, model_21, x_1d, 0.1,2).detach().numpy()