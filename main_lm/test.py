import torch

def real_solution_1d(x):
    return torch.sin(x)

class nn_params_options:
    def __init__(self,input_dim,n_hidden_layers,r_nodes_per_layer,output_dim=1,activation_function=torch.sigmoid,sample_num=40):
        self.input_dim = input_dim
        self.n_hidden_layers = n_hidden_layers
        self.r_nodes_per_layer = r_nodes_per_layer
        self.output_dim = 1
        self.activation_function = torch.sigmoid
        self.sample_num = 40 #in 1d, i.e. input_data = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
        

class LMTR_params_options:
    def __init__(self,eta1=0.1,eta2=0.75,gamma1=0.85,gamma2=0.5,gamma3=1.5,lambda_min=1e-4,epsilon = 1e-4,max_iter=1000):
        self.eta1 = 0.1 #pho successful 
        self.eta2 = 0.75 #pho very successful
        self.gamma1 = 0.85 #step is successful but not very successful,shrink the regularization coefficient (lambda0)
        self.gamma2 = 0.5 #step is very successful, shrink the regularization coefficient(lambda0)
        self.gamma3 = 1.5 #step failed, increase regularization coefficient(lambda0)
        #self.lambdak = 0.05 #initial value of the regularization coefficient
        self.lambda_min = 1e-4 #the minimum of the regularization coefficient
        self.epsilon = 1e-4 #the tolerance of grad_obj
        self.max_iter = 1000 # the maximum of the number of iterations
        
        assert 0<eta1<=eta2<1
        assert 0<gamma2<=gamma1<1<gamma3
        assert lambda_min>0
        assert epsilon>0 
        

from Multilevel_LM.main_lm.LMTR_poisson import LMTR_solving_poisson
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
pred_1d = LMTR_solving_poisson(test_func_1d, model_21, x_1d, lambdak=0.1).detach().numpy()
real_1d = test_func_1d(x_1d).detach().numpy()
plt.figure(figsize = (10,5))
x_1d_np = x_1d.detach().numpy()
plt.plot(x_1d_np,pred_1d,label = 'Model Output',color = 'blue',linestyle='-',marker = 'o',markersize=2)
plt.plot(x_1d_np, real_1d, label='Real Solution', color='red', linestyle='dashed')
plt.title('Plot of Model Output vs. Input x')
plt.xlabel('Input x')
plt.ylabel('Model Output')
plt.legend()
plt.grid()
plt.show()



  
    
