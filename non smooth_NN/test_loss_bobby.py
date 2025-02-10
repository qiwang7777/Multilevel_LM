#In this file, we generate the dataset, input dimension is 2 (spatial data in 2d) and output dimension is 1
#To get all the values we need to put into loss function, including the value of kappa and f.
from fenics import *
from dolfin import *

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


import matplotlib.pyplot as plt


# Fully connected neural network construction
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid() #changable

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def grad_nn(model,inputs):

    inputs = inputs.clone().detach().requires_grad_(True)

    outputs = model(inputs).squeeze(0)


    if outputs.shape[1] != 1:
        raise ValueError("The output of model must be a scalar (shape [batch_size,1]).")
    gradients = torch.autograd.grad(outputs,inputs,grad_outputs=torch.ones_like(outputs),create_graph = True)[0]
    return gradients

def hessian_nn(model,inputs):
    inputs = inputs.clone().detach().requires_grad_(True)
    inputs_num = inputs.shape[0]
    hessian = torch.zeros(inputs_num,2,2,dtype=torch.float32)
    outputs = model(inputs)
    gradients = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]
    for i in range(2):
        grad_i = gradients[:, i]
        # Calculate the second-order derivatives with respect to the inputs
        second_gradients = torch.autograd.grad(grad_i, inputs, grad_outputs=torch.ones_like(grad_i), create_graph=True)[0]
        # Store the second-order derivatives in the Hessian matrix
        hessian[:, i, :] = second_gradients

    return hessian

def laplacian_nn(model,inputs):
    hessian = hessian_nn(model,inputs)
    laplacian = hessian[:, 0, 0] + hessian[:, 1, 1]
    return laplacian


#inputs_domain = generate_fenics_data(n_samples=10,meshsize=4)[0]

class PDEObjective:
    def __init__(self, n_samples = 10, meshsize = 31, reg_param = 0.01):
        self.mshsize   = meshsize
        self.nsamps    = n_samples
        self.reg_param = reg_param
                # Mesh grid for x, y ((meshsize+1) x (meshsize+1))
        self.mesh = UnitSquareMesh(meshsize, meshsize)
        self.V = FunctionSpace(self.mesh, "Lagrange", 1)

        # Boundary condition
        self.u_D = Constant(0.0)
        self.bc = DirichletBC(self.V, self.u_D, "on_boundary")

        # Create vertex coordinates for mapping
        vertex_coords = self.mesh.coordinates()
        self.x_coords = vertex_coords[:, 0]

        self.y_coords = vertex_coords[:, 1]

        # Define masks for boundary and domain
        self.boundary_mask = (np.isclose(self.x_coords, 0) | np.isclose(self.x_coords, 1) |
                        np.isclose(self.y_coords, 0) | np.isclose(self.y_coords, 1))
        self.domain_mask   = ~self.boundary_mask

        t = self.input_data()
        self.inputs_domain   = t[0]
        self.inputs_boundary = t[1]
        self.inputs          = t[2]

        t = self.generate_fenics_data()
        self.kappa_value_list  = t[0]
        self.f_value_list      = t[1]
        self.kappa_grad_list   = t[2]
        self.u_solution_tensor = t[3]

    def input_data(self):
        # mesh = UnitSquareMesh(meshsize, meshsize)
        # vertex_coords = self.mesh.coordinates()
        # x_coords = vertex_coords[:,0]
        # y_coords = vertex_coords[:,1]
        # boundary_mask = (np.isclose(x_coords, 0) | np.isclose(x_coords, 1) |
                        # np.isclose(y_coords, 0) | np.isclose(y_coords, 1))
        # domain_mask = ~boundary_mask
        inputs = []
        inputs_features = np.stack([self.x_coords,self.y_coords], axis=-1)
        inputs.append(inputs_features)
        inputs = torch.tensor(np.array(inputs), dtype=torch.float32).squeeze(0)
        inputs_domain = []
        input_domain_features = np.stack([self.x_coords[self.domain_mask],self.y_coords[self.domain_mask]], axis=-1)
        inputs_domain.append(input_domain_features)
        inputs_domain = torch.tensor(np.array(inputs_domain), dtype=torch.float32).squeeze(0)
        inputs_boundary = []
        input_boundary_features = np.stack([self.x_coords[self.boundary_mask],self.y_coords[self.boundary_mask]], axis=-1)
        inputs_boundary.append(input_boundary_features)
        inputs_boundary = torch.tensor(np.array(inputs_boundary), dtype=torch.float32).squeeze(0)
        return inputs_domain,inputs_boundary,inputs

    def generate_fenics_data(self):
        # Generate random values for kx, ky, ax, ay, and alpha
        kx_samples    = np.random.uniform(0.5, 4.0, self.nsamps)
        ky_samples    = np.random.uniform(0.5, 4.0, self.nsamps)
        ax_samples    = np.random.uniform(0.0, 0.5, self.nsamps)
        ay_samples    = np.random.uniform(0.0, 0.5, self.nsamps)
        alpha_samples = np.random.uniform(0.0, np.pi / 2, self.nsamps)
        print('up here')

        # Source term
        f = Expression("32*exp(-4*((x[0]-0.25)*(x[0]-0.25) + (x[1]-0.25)*(x[1]-0.25)))", degree=2)


        # Prepare containers for inputs and outputs
        kappa_value_list = []
        f_value_list = []
        kappa_grad_list = []
        u_solution_list = []


        for i in range(self.nsamps):
            kx = kx_samples[i]
            ky = ky_samples[i]
            ax = ax_samples[i]
            ay = ay_samples[i]
            alpha = alpha_samples[i]

            # Define rotated coordinates x' and y'
            kappa = Expression(
                "1.1 + cos(kx*pi*(cos(alpha)*(x[0]-0.5) - sin(alpha)*(x[1]-0.5) + 0.5 + ax)) * cos(ky*pi*(sin(alpha)*(x[0]-0.5) + cos(alpha)*(x[1]-0.5) + 0.5 + ay))",
                degree=2, kx=kx, ky=ky, ax=ax, ay=ay, alpha=alpha, pi=np.pi
            )
            # Define the gradient as a vector function
            kappa_func = project(kappa,self.V)
            vector_space = VectorFunctionSpace(self.mesh, "CG", 1)
            # t = as_vector([kappa_func.dx(0), kappa_func.dx(1)])
            # print(t)
            # grad_kappa = project(t, vector_space)
            kappa_x = Expression(
                "kx*pi*alpha*sin(kx*pi*(cos(alpha)*(x[0]-0.5) - sin(alpha)*(x[1]-0.5) + 0.5 + ax))*"
                "cos(ky*pi*(sin(alpha)*(x[0]-0.5) + cos(alpha)*(x[1]-0.5) + 0.5 + ay))*sin(alpha*(x[0]-0.5))-"
                "ky*pi*alpha*cos(kx*pi*(cos(alpha)*(x[0]-0.5) - sin(alpha)*(x[1]-0.5) + 0.5 + ax))*"
                "sin(ky*pi*(sin(alpha)*(x[0]-0.5) + cos(alpha)*(x[1]-0.5) + 0.5 + ay))*cos(alpha*(x[0]-0.5))",
                degree=2, kx=kx, ky=ky, ax=ax, ay=ay, alpha=alpha, pi=np.pi
                )

            kappa_y = Expression(
                "kx*pi*alpha*sin(kx*pi*(cos(alpha)*(x[0]-0.5) - sin(alpha)*(x[1]-0.5) + 0.5 + ax))*"
                "cos(ky*pi*(sin(alpha)*(x[0]-0.5) + cos(alpha)*(x[1]-0.5) + 0.5 + ay))*cos(alpha*(x[1]-0.5))+"
                "ky*pi*alpha*cos(kx*pi*(cos(alpha)*(x[0]-0.5) - sin(alpha)*(x[1]-0.5) + 0.5 + ax))*"
                "sin(ky*pi*(sin(alpha)*(x[0]-0.5) + cos(alpha)*(x[1]-0.5) + 0.5 + ay))*sin(alpha*(x[1]-0.5))",
            degree=2, kx=kx, ky=ky, ax=ax, ay=ay, alpha=alpha, pi=np.pi
            )

            kappa_x_func = project(kappa_x,self.V)
            kappa_y_func = project(kappa_y,self.V)
            vector_space = VectorFunctionSpace(self.mesh,"CG",1)
            kappa_grad = project(as_vector([kappa_x_func,kappa_y_func]),vector_space)


            # Define variational problem
            u = TrialFunction(self.V)
            v = TestFunction(self.V)
            a = dot(kappa * grad(u), grad(v)) * dx
            L = f * v * dx

            # Solve the problem
            u_sol = Function(self.V)
            solve(a == L, u_sol, self.bc)
            # Get the solution and input features
            u_array           = u_sol.compute_vertex_values(self.mesh)
            kappa_values      = kappa.compute_vertex_values(self.mesh)
            #grad_kappa_values = grad_kappa.compute_vertex_values(mesh).reshape(-1, 2)
            grad_kappa_values = kappa_grad.compute_vertex_values(self.mesh).reshape(-1, 2)


            f_values          = f.compute_vertex_values(self.mesh)
            kappa_domain      = kappa_values[self.domain_mask]
            f_domain          = f_values[self.domain_mask]
            grad_kappa_domain = grad_kappa_values[self.domain_mask,:]

            kappa_value_list.append(kappa_domain)
            f_value_list.append(f_domain)
            kappa_grad_list.append(grad_kappa_domain)
            u_solution_list.append(u_array)


        kappa_grad_list = torch.tensor(np.array(kappa_grad_list), dtype=torch.float32)
        kappa_value_list = torch.tensor(np.array(kappa_value_list), dtype=torch.float32)
        f_value_list= torch.tensor(np.array(f_value_list), dtype=torch.float32)
        u_solution_tensor = torch.tensor(np.array(u_solution_list),dtype=torch.float32)


        return  kappa_value_list, f_value_list, kappa_grad_list, u_solution_tensor
        #Size of the results from function generate_fenics_data
        #inputs_domain : spatial data (x,y) with size of [1,(meshsize-1)**2,2], there are (meshsize-1)**2 points in interior and the spatial data is in 2d
        #outputs_domain: u(x,y) when (x,y) is in (0,1)^2, whose size is [n_samples,(meshsize-1)**2,1] as kappa would change with the different samples
        #inputs_boundary: spatial data (x,y) with size of [n_samples,(meshsize+1)**2-(meshsize-1)**2,2], there are (meshsize+1)**2-(meshsize-1)**2 points in boundary
        #outputs_boundary: u(x,y) when (x,y) in on the boundary, whose size is [n_samples,(meshsize+1)**2-(meshsize-1)**2,1]
        #kappa_value_list: the value of kappa(kx,ky,ax,ay,alpha,x,y), here only consider when (x,y) is in (0,1)^2
        #f_value_list: the value of f(x,y), here only consider when (x,y) is in (0,1)^2


    def loss(self, model):
        # print('in loss')
        # inputs_domain = input_data(meshsize)[0]
        inputs_domain = self.inputs_domain.requires_grad_()
        # inputs_boundary = input_data(meshsize)[1]

        # nn_domain = model(inputs_domain)
        nn_boundary = model(self.inputs_boundary)
        #pde residual -kappa*laplacian_nn-dot(grad_kappa,grad_nn)-f
        nn_grad = grad_nn(model, inputs_domain)
        # kappa_grad = generate_fenics_data(n_samples,meshsize)[2]
        kappa_grad         = self.kappa_grad_list
        grad_kappa_grad_nn = torch.einsum('ijk,jk->ij',kappa_grad, nn_grad)


        # kappa_domain = generate_fenics_data(n_samples, meshsize)[0]
        kappa_domain = self.kappa_value_list
        nn_laplacian = laplacian_nn(model,self.inputs_domain)
        kappa_laplacian_nn = kappa_domain*nn_laplacian


        # f_domain = generate_fenics_data(n_samples, meshsize)[1]
        f_domain = self.f_value_list

        pde_residual = -grad_kappa_grad_nn - kappa_laplacian_nn - f_domain


        loss_pde = torch.mean(pde_residual**2)

        #boundary residual

        loss_boundary = torch.mean(nn_boundary**2)

        #regularization
        loss_reg = self.reg_param * sum(param.abs().sum() for param in model.parameters())

        loss = loss_pde+loss_boundary+loss_reg


        return loss

# Training Loop
def train_model(model,optimizer,obj):

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        # loss = loss_pde(model,n_samples=10,meshsize = 31,reg_param=0.01)
        loss = obj.loss(model)
        data_num = obj.nsamps*(obj.mshsize+1)*(obj.mshsize+1)

            # Forward pass
            #outputs = model(inputs)
            #targets = targets.view(-1, 32, 32)
            #loss = criterion(outputs, targets)

            # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss/data_num:.4f}")


def plot_comparison(model, PDEObj, n_samples=1):
    # Generate data for a single sample
    # inputs_domain, inputs_boundary, inputs = PDEObj.input_data(meshsize)
    meshsize = PDEObj.mshsize
    real_solution = PDEObj.u_solution_tensor
    # real_solution = PDEObj.generate_fenics_data(n_samples=n_samples, meshsize=meshsize)[3]

    # Pass inputs_domain through the trained model to get predictions
    predicted_solution = model(PDEObj.inputs_domain).detach().numpy()
    # Convert solutions into 2D grids
    # real_solution_grid = real_solution.squeeze(0).reshape(meshsize + 1, meshsize + 1).numpy()
    real_solution_grid = real_solution[0].reshape(meshsize + 1, meshsize + 1).numpy()
    predicted_solution_grid = predicted_solution.reshape(meshsize - 1, meshsize - 1) #not correct?

    # Compute error (absolute difference)
    error_grid = abs(real_solution_grid[1:31, 1:31] - predicted_solution_grid)

    # Plot real solution, predicted solution, and error
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Real solution
    ax = axes[0]
    im = ax.imshow(real_solution_grid, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    ax.set_title("Real Solution (FEniCS)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

    # Predicted solution
    ax = axes[1]
    im = ax.imshow(predicted_solution_grid, extent=[0, 1, 0, 1], origin="lower", cmap="viridis")
    ax.set_title("Predicted Solution (NN)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

    # Error
    ax = axes[2]
    im = ax.imshow(error_grid, extent=[0, 1, 0, 1], origin="lower", cmap="plasma")
    ax.set_title("Error (|Real - Predicted|)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


# Main Script
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 2
    output_dim = 1
    hidden_dim = 400
    epochs = 450
    # model = FullyConnectedNN(input_dim, hidden_dim, output_dim)
    #Loss function
    #PDE residual, for the pde residual part, we need the laplacian_nn and also the gradient of nn

    # Initialize the model, loss, and optimizer
    model     = FullyConnectedNN(input_dim, hidden_dim, output_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    PDEObj    = PDEObjective(n_samples=10,meshsize=31,reg_param=0.0)



    # Train the model
    train_model(model,optimizer, PDEObj)
    # Save the model
    torch.save(model.state_dict(), "fully_connected_nn.pth")
    print("Model training complete and saved!")
    # Example usage after training the model
    plot_comparison(model, PDEObj, n_samples = 1)





