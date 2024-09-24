import torch


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
#2d test
input_dim_2d = 2
output_dim_2d = 1
n_hidden_layers_2d = 1
r_nodes_per_layer_2d = 100
model_21_2d = FullyConnectedNN(input_dim_2d, n_hidden_layers_2d, r_nodes_per_layer_2d, output_dim_2d)


x = np.linspace(0, 1, 41)
y = np.linspace(0, 1, 41)

# Create a 2D grid
X, Y = np.meshgrid(x, y)

# Flatten the grid and stack x and y coordinates
input_data = np.stack([X.flatten(), Y.flatten()], axis=1)
def test_func_2d(x):
    return torch.sin(x[:,0])+x[:,1]
# Convert to PyTorch tensor
x_2d = torch.tensor(input_data, dtype=torch.float32)
pred_2d = LMTR_solving_poisson(test_func_2d, model_21_2d, x_2d, lambdak=0.1).detach().numpy()
real_2d = test_func_2d(x_2d).detach().numpy()
real_2d_reshaped = real_2d.reshape(X.shape)

from mpl_toolkits.mplot3d import Axes3D
pred_2d_reshaped = pred_2d.reshape(X.shape)
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(111,projection='3d')
surf_model = ax.plot_surface(X,Y,pred_2d_reshaped,cmap = 'viridis',alpha = 0.7)
surf_real = ax.plot_surface(X,Y,real_2d_reshaped,cmap = 'plasma',alpha = 0.5)
ax.set_title('3D Plot of Model Output and Real Solution')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Model Output')
fig.colorbar(surf_model, ax=ax, shrink=0.5, aspect=10)
plt.show()
    



  
    
