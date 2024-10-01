#To get the prolongation operators and restriction operators, 
#we need to get some matrix A first to give us the sufficient information.
#First, get the matrix A, where A is related to the partial derivative of loss function w.r.t. w,v,b

from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
from Multilevel_LM.main_lm.PoissonPDE import PoissonPDE
from Multilevel_LM.main_lm.loss_poisson import loss_solving_poisson
import torch,copy
import numpy as np

def compute_loss_gradients_partial(real_solution,model,x,regularization=True,lambdap = 0.1):
    #Forward pass: Perform forward pass to compute network output
    model.zero_grad()
    total_loss = loss_solving_poisson(real_solution,model,x,regularization=True,lambdap = 0.1)
    total_loss.backward()
    gradients = []
    for param in model.parameters():
        gradients.append(param.grad)
    
    
    return gradients

def information_A(real_solution,model,x,regularization=True,lambdap = 0.1):
    input_dim = x.shape[1]
    r = model.r_nodes_per_layer
    grad = compute_loss_gradients_partial(real_solution,model,x,regularization=True,lambdap = 0.1)
    if input_dim == 1:
        
        Aw = grad[0].view(r,1)
        Ab = grad[1].view(r,1)
        Av = grad[2].view(r,1)
        return Aw@Aw.T/np.linalg.norm(Aw,ord = np.inf)+Ab@Ab.T/np.linalg.norm(Ab,ord = np.inf)+Av@Av.T/np.linalg.norm(Av,ord = np.inf)

    else:
        Aw1 = grad[0][:,0].view(r,1)
        
        Aw2 = grad[0][:,1].view(r,1)
       
        Ab = grad[1].view(r,1)
        
        Av = grad[2].view(r,1)
        
        return Aw1@Aw1.T/np.linalg.norm(Aw1,ord = np.inf)+Aw2@Aw2.T/np.linalg.norm(Aw2,ord = np.inf)+Ab@Ab.T/np.linalg.norm(Ab,ord = np.inf)+Av@Av.T/np.linalg.norm(Av,ord = np.inf)
    
def in_A(real_solution,model,x,regularization=True,lambdap = 0.1):
    input_dim = x.shape[1]
    r = model.r_nodes_per_layer
    grad = compute_loss_gradients_partial(real_solution,model,x,regularization=True,lambdap = 0.1)
    inform_A = torch.zeros(r,r)
    if input_dim == 1:
        for i in range(len(grad)-1):
            Ai = grad[i].view(r,1)
            inform_A += Ai@Ai.mT/np.linalg.norm(Ai,ord = np.inf)
        #return inform_A
    else:
        for i in range(len(grad)-1):
            Ai = grad[i]
            #print(Ai)
            if min(Ai.shape) == 1 or min(Ai.shape)==max(Ai.shape):
                Ai = Ai.view(r,1)
                inform_A += Ai@Ai.T/np.linalg.norm(Ai,ord = np.inf)
            else: 
                for j in range(min(Ai.shape)):
                    #print(Ai[:,j].view(r,1)@Ai[:,j].view(r,1).mT)
                    Aij = Ai[:,j].view(r,1)
                    #print(Aij)
                    inform_A += Aij@Aij.T/np.linalg.norm(Aij,ord=np.inf)
        #return inform_A
    return inform_A

#try AMG 

#If |A[i,j]| >= theta * max|A[i,k]| while k!=i, then we call j is strongly connected to i.
#Note that A is symmetric matrix

    
    
from pyamg.classical.split import RS
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.interpolate import direct_interpolation
from scipy.sparse import csr_matrix             
from pyamg.amg_core import rs_cf_splitting    
import pyamg
from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN
input_dim = 1
output_dim = 1
n_hidden_layers = 1
r_nodes_per_layer = 5
model_21 = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim)
sample_num = 5
x_1d = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,1), dtype=torch.float32)

def test_func_1d(x):
    return torch.sin(x)


#print(compute_loss_gradients_partial(test_func_1d,model_21,x_1d,regularization=True,lambdap = 0.1))
#in_A_1d = in_A(test_func_1d,model_21,x_1d,regularization=True,lambdap = 0.1)
#torch.save(in_A_1d,'A.pt')
loaded_tensor=torch.load('A.pt')
print(loaded_tensor)
#A = np.array([[-5.5, -6.0, 3.0, -10.0, -9.8, 5.0, 4.0, 2.0, 1.0, 0.0],
#              [-6.0, 4.0, 0.0, -7.0, -6.5, 3.0, 0.0, -2.0, 0.0, 1.0],
#              [3.0, 0.0, -4.5, -6.5, -7.0, 0.0, 4.0, 1.0, 0.0, 0.0],
#              [-10.0, -7.0, -6.5, 2.0, -9.2, 6.0, 5.0, 9.8, 10.0, 10.0],
#              [-9.8, -6.5, -7.0, -9.2, 4.0, 0.0, 2.0, 10.0, -10.0, 1.0],
#              [5.0, 3.0, 0.0, 6.0, 0.0, 5.5, -5.0, -4.8, 7.0, 8.0],
#              [4.0, 0.0, 4.0, 5.0, 2.0, -5.0, 1.0, 0.0, 4.0, -4.9],
#              [2.0, -2.0, 1.0, 9.8, 10.0, -4.8, 0.0, 2.0, -5.0, 7.0],
#              [1.0, 0.0, 0.0, 10.0, -10.0, 7.0, 4.0, -5.0, 4.8, -9.9],
#              [0.0, 1.0, 0.0, 10.0, 1.0, 8.0, -4.9, 7.0, -9.9, 5.0]])
#in_A_1d = torch.tensor(A)
#print(in_A_1d)
ml_A_1d = pyamg.classical.ruge_stuben_solver(loaded_tensor,strength=('classical',{'theta':0.25}),max_levels = 2, max_coarse=1, CF='RS',keep = True)
print(ml_A_1d)
splitting_A_1d = ml_A_1d.levels[0].splitting
C_nodes = splitting_A_1d  == 1
F_nodes = splitting_A_1d  == 0
print(C_nodes)
print(F_nodes)
#result is not good, because even we chose 500 hundreds nodes in the hidden layer, and a small connected parameter theta 
#i.e. |A[i,j]| >= theta * max|A[i,k]| while k!=i, then we call j is strongly connected to i.
#C_nodes might be a single set, which means only one point in C set, the reason should be from the choice of information
#matrix A. According to this, we might need change the connected parameters with the change of matrix A. Furthermore, 
#if the cardinality of C set is too small, which is not good for construnction of neural network.







#input_dim_2d = 2
#output_dim_2d = 1
#n_hidden_layers_2d = 1
#r_nodes_per_layer_2d = 100
#model_21_2d = FullyConnectedNN(input_dim_2d, n_hidden_layers_2d, r_nodes_per_layer_2d, output_dim_2d)


#x = np.linspace(0, 1, 3)
#y = np.linspace(0, 1, 3)

# Create a 2D grid
#X, Y = np.meshgrid(x, y)

# Flatten the grid and stack x and y coordinates
#input_data = np.stack([X.flatten(), Y.flatten()], axis=1)
#def test_func_2d(x):
#    return torch.sin(x[:,0])+x[:,1]
# Convert to PyTorch tensor
#x_2d = torch.tensor(input_data, dtype=torch.float32)

#print(information_A(test_func_2d,model_21_2d,x_2d)==in_A(test_func_2d,model_21_2d,x_2d))
