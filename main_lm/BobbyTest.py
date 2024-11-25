
import sys, os
os.chdir("/home/rjbaral/Documents/Research/trustedAI/training/code/python")
sys.path.append("/home/rjbaral/Documents/Research/trustedAI/training/code/python")
import numpy as np 


from TorchObjectives import TrainingObjective, SquaredErrorObjective
from TorchTrainer    import TorchTrainer
from TorchVectors    import TensorDictVector

import torch
import torch.nn.functional as F
from pyrol import getCout, Objective, Problem, Solver
from pyrol.pyrol.Teuchos import ParameterList

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import time

class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function=nn.Sigmoid()):
        super(FullyConnectedNN, self).__init__()
        self.input_dim         = input_dim
        self.n_hidden_layers   = n_hidden_layers
        self.r_nodes_per_layer = r_nodes_per_layer
        self.output_dim        = output_dim
        
        
        self.hidden_layers = nn.ModuleList()
        self.activation_function = activation_function
        
        # Input layer (first hidden layer)
        self.hidden_layers.append(nn.Linear(input_dim, r_nodes_per_layer))
        
        # Additional hidden layers
        for _ in range(1, n_hidden_layers):
            self.hidden_layers.append(nn.Linear(r_nodes_per_layer, r_nodes_per_layer))
        
        # Output layer
        self.output_layer = nn.Linear(r_nodes_per_layer, output_dim)
    
    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.activation_function(layer(x))
        x = self.output_layer(x)
        return x

class MSEObjective(SquaredErrorObjective):
    # TO-DO: Pass torch.nn.MSELoss(reduction='mean') to TrainingObjective.

    def __init__(self, model, data):
        super().__init__(model, data)

    def torch_value(self, x):
        return super().torch_value(x)/self.x.shape[0]

def create_parameter_list():
    p = ParameterList()
    p['General'] = ParameterList()
    p['General']['Output Level'] = 1
    p['Step'] = ParameterList()
    p['Step']['Trust Region'] = ParameterList()
    p['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
    #p['Step']['Type'] = 'Line Search'
    #p['Step']['Line Search'] = ParameterList()
    #p['Step']['Line Search']['Initial Step Size'] = 1e6
    #p['Step']['Line Search']['Line-Search Method'] = ParameterList()
    #p['Step']['Line Search']['Line-Search Method']['Type'] = 'Bisection'
    p['Step']['Descent Method'] = ParameterList()
    p['Step']['Descent Method']['Type'] = 'Nonlinear CG'
    p['Step']['Trust Region']['SPG'] = ParameterList()
    p['Step']['Trust Region']['SPG']['Solver'] = ParameterList()
    p['Step']['Trust Region']['SPG']['Solver']['Iteration Limit'] = 10

    p['Status Test'] =  ParameterList()
    p['Status Test']['Gradient Tolerance'] = 1e-4
    p['Status Test']['Use Relative Tolerances'] = True 
    p['Status Test']['Iteration Limit'] = 100
    return p

def main():
   indim  = 1
   outdim = 1
   nhidim = 1
   rnode  = 300
   ModelF = FullyConnectedNN(indim, nhidim, rnode, outdim)
   ModelF.double()
   x      = TensorDictVector(ModelF.state_dict())
   #x_1d   = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,1), dtype=torch.float32)
   xin    = torch.tensor(np.linspace(0, 1, 1), dtype = torch.double)
   xout   = ModelF(xin)
   
   obj    = MSEObjective(ModelF, (xin, xout))

   g      = x.dual()

   problem = Problem(obj, x, g)
   stream  = getCout()
   p       = create_parameter_list()
   solver  = Solver(problem, p)

   problem.checkDerivatives(True, stream)

if __name__ == "__main__":
    main()
