

from ROL import Objective

import torch



class TorchObjective(Objective):
    # https://pytorch.org/docs/stable/func.html

    # @staticmethod
    # def _copy(source, target):
    #     target.zero()
    #     target.plus(source)

    def __init__(self):
        super().__init__()
        self.torch_gradient = torch.func.grad(self.torch_value)

    def value(self, x, tol):
        return self.torch_value(x).item()

    def torch_value(self, x):
        # Returns a scalar torch Tensor
        raise NotImplementedError

    def gradient(self, g, x, tol):
        ans = self.torch_gradient(x)
        g.copy_(ans) 

    def _forward_over_reverse(self, input, x, v):
        # https://github.com/google/jax/blob/main/docs/notebooks/autodiff_cookbook.ipynb
        return torch.func.jvp(input, (x,), (v,))

    def hessVec(self, hv, v, x, tol):
        _, ans = self._forward_over_reverse(self.torch_gradient, x, v)
        hv.copy_(ans)


class SquaredNorm(TorchObjective):

    def torch_value(self, x):
        return 0.5 * torch.sum(x.squeeze() ** 2)


class TrainingObjective(TorchObjective):

    def __init__(self, model, data, loss):
        super().__init__()
        self.model = model
        self.x, self.y = data
        self.loss = loss

    def torch_value(self, x):
        return self.loss(torch.func.functional_call(self.model, x, self.x), self.y)


class SquaredErrorObjective(TrainingObjective):

    def __init__(self, model, data):
        loss = torch.nn.MSELoss(reduction='sum')
        super().__init__(model, data, loss)
        
#Test for calculating Jacobian and Hessian
#class MyFuncObjective(TorchObjective):

#    def torch_value(self, x):
        # Assuming x is a torch tensor containing [x, y]
#        z, y = x[0], x[1]
#        return z**2 + 2*z*y + y**2


# Test Example
#x = torch.tensor([1.0, 2.0], requires_grad=True)
#v = torch.tensor([1.0, 0.0])
#hv = torch.tensor([0.0, 0.0])

#obj = MyFuncObjective()
#tolerance = 1e-6

# Calculate value
#value = obj.value(x, tolerance)
#print(f"Objective Value: {value}")

# Calculate gradient
#g = torch.zeros_like(x)  # Initialize g with the same shape as x
#obj.gradient(g, x, tolerance)
#print(f"Gradient: {g}")

# Calculate Hessian-vector product
#obj.hessVec(hv, v, x, tolerance)
#print(f"Hessian-Vector Product: {hv}")

#Usage for Nxx
#import numpy as np
#import Multilevel_LM
#from Multilevel_LM.main_lm.neural_network_construction import FullyConnectedNN

#input_dim = 1
#output_dim = 1
#n_hidden_layers = 1
#r_nodes_per_layer = 500
#activation_function = torch.sigmoid

# Create the network
#My_model_nn = FullyConnectedNN(input_dim, n_hidden_layers, r_nodes_per_layer, output_dim, activation_function)



#class NNObjective(TorchObjective):
    
#    def torch_value(self,x):
#        return torch.sum(My_model_nn(x))
    
#Test Example
#sample_num = 3
#xx = torch.tensor(np.linspace(0,1,sample_num).reshape(sample_num,input_dim), dtype=torch.float32)
#value = torch.zeros_like(xx)
#gradient = torch.zeros_like(xx)
#hessian = torch.zeros_like(xx)
#for i in range(sample_num):
#    hv = torch.tensor([0.0])
#    v = torch.tensor([1.0]) #v could also be -1.0
#    xx_i = xx[i]
#    obj_nn = NNObjective()
#    tolerance = 1e-6
    #Calculte the value
    
#    value[i] = obj_nn.value(xx_i,tolerance)
    #print(f"Ouput of NN with {xx_i}: {value[i]}")
    
#    g_i = torch.zeros_like(xx_i)
   
#    obj_nn.gradient(g_i, xx_i, tolerance)
#    gradient[i] = g_i
    
    #print(f"Gradient of NN with repect to {xx_i} :{g_i}")
    
#    obj_nn.hessVec(hv, v, xx_i, tolerance)
#    hessian[i] = hv
    #print(f"Hessian-Vector Product of NN w.r.t. {xx_i}: {hv}")
#print(gradient)
#print(hessian)



#Test for objective loss function


#Calculte the gradient and second derivative of NN w.r.t. input x
def compute_first_derivative(model,input_data):
    input_data.requires_grad = True

    # Forward pass: Compute the network output
    output = model(input_data)

    # First derivative: Compute the gradient of the output with respect to the input
    first_derivative = torch.autograd.grad(outputs=output, inputs=input_data,
                                           grad_outputs=torch.ones_like(output),
                                           create_graph=True, allow_unused=True)[0]
    return first_derivative
    
#Test for function compute_first_derivative
#print(compute_first_derivative(My_model, input_data))
#it works
def compute_second_derivative(model, input_data):
    input_data.requires_grad = True


    # First derivative: Compute the gradient of the output with respect to the input
    first_derivative = compute_first_derivative(model, input_data)

    if first_derivative is None:
        return ValueError("First derivative is None. Check if the input tensor is used in the graph.")


    # Second derivative: Compute the gradient of the first derivative with respect to the input
    second_derivative = torch.zeros_like(input_data)
    for i in range(input_data.size(0)):  # Iterate over input features
        grad2 = torch.autograd.grad(first_derivative[i], input_data, retain_graph=True, allow_unused=True)[0]
        if grad2 is not None:
            second_derivative[i] = grad2[i]

    return second_derivative
    


#print(compute_first_derivative(My_model_nn, xx)) #same with the above methods
#print(compute_second_derivative(My_model_nn, xx)) #same with the above methods  

