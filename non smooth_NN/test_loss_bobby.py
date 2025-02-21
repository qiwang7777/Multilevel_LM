#In this file, we generate the dataset, input dimension is 2 (spatial data in 2d) and output dimension is 1
#To get all the values we need to put into loss function, including the value of kappa and f.
import ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, set_bc

import numpy as np

import torch
import torch.nn as nn
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

class PDEObjective:
    def __init__(self, n_samples = 10, meshsize = 31, reg_param = 0.01):
        self.mshsize   = meshsize
        self.nsamps    = n_samples
        self.reg_param = reg_param
                # Mesh grid for x, y ((meshsize+1) x (meshsize+1))
        self.mesh = UnitSquareMesh(meshsize, meshsize)
        self.V = FunctionSpace(self.mesh, "Lagrange", 1)
        self.vs = VectorFunctionSpace(self.mesh, "CG", 1)
        # Source term
        self.f = Expression("32*exp(-4*((x[0]-0.25)*(x[0]-0.25) + (x[1]-0.25)*(x[1]-0.25)))", degree=2)

        # u = project(Expression(("x[0]", "x[1]"), degree = 1), self.vs) #get coordinates?
        # T = TensorFunctionSpace(self.mesh, "DG", 0) #i think gradient is in tensor space 1 deg lower than originalvs
        # g = project(grad(u), T)
        # self.grad = torch.from_numpy(g.vector().get_local())
        # print(self.grad.size())

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
        # self.kappa_grad_list   = t[2]
        self.u_solution_tensor = t[2]

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


        # Prepare containers for inputs and outputs
        kappa_value_list = []
        f_value_list = []
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
            self.kappa = kappa

            V = self.V
            # Define variational problem
            u = TrialFunction(V)
            v = TestFunction(V)
            # a = dot(kappa * grad(u), grad(v)) * dx
            a = dot(kappa * grad(u), grad(v)) * dx
            L = self.f * v * dx

            dxp = dx(metadata = {"quadrature_scheme":"vertex"})
            A = dot(kappa*grad(u), grad(v))*dxp #kappa is throwing error
            A = Form(A, bcs = self.bc)
            A = assemble(A)
            # print(A.array())
            A = A.array()
            A = A.astype(np.float32)
            self.A = torch.from_numpy(A)
            # Solve the problem
            u_sol = Function(self.V)
            solve(a == L, u_sol, self.bc)
            # Get the solution and input features
            u_array           = u_sol.compute_vertex_values(self.mesh)
            kappa_values      = kappa.compute_vertex_values(self.mesh)

            f_values          = self.f.compute_vertex_values(self.mesh)
            print(np.dot(A, u_array) - f_values)

            kappa_domain      = kappa_values[self.domain_mask]
            f_domain          = f_values[self.domain_mask]

            kappa_value_list.append(kappa_domain)
            f_value_list.append(f_domain)
            u_solution_list.append(u_array)


        # kappa_grad_list = torch.tensor(np.array(kappa_grad_list), dtype=torch.float32)
        kappa_value_list = torch.tensor(np.array(kappa_value_list), dtype=torch.float32)
        f_value_list= torch.tensor(np.array(f_value_list), dtype=torch.float32)
        u_solution_tensor = torch.tensor(np.array(u_solution_list),dtype=torch.float32)


        return  kappa_value_list, f_value_list, u_solution_tensor
        #Size of the results from function generate_fenics_data
        #inputs_domain : spatial data (x,y) with size of [1,(meshsize-1)**2,2], there are (meshsize-1)**2 points in interior and the spatial data is in 2d
        #outputs_domain: u(x,y) when (x,y) is in (0,1)^2, whose size is [n_samples,(meshsize-1)**2,1] as kappa would change with the different samples
        #inputs_boundary: spatial data (x,y) with size of [n_samples,(meshsize+1)**2-(meshsize-1)**2,2], there are (meshsize+1)**2-(meshsize-1)**2 points in boundary
        #outputs_boundary: u(x,y) when (x,y) in on the boundary, whose size is [n_samples,(meshsize+1)**2-(meshsize-1)**2,1]
        #kappa_value_list: the value of kappa(kx,ky,ax,ay,alpha,x,y), here only consider when (x,y) is in (0,1)^2
        #f_value_list: the value of f(x,y), here only consider when (x,y) is in (0,1)^2

    def loss(self, x):
        #x is the parameters of model
        #print(f"before change: {x.shape}")
        nn = torch.func.functional_call(self.model, x, (self.inputs,))

        f_domain = self.f_value_list
        pde = torch.matmul(self.A,nn)
        #I suppose we can also domain_mask A?
        pde_residual = pde[self.domain_mask] - f_domain

        loss_pde = torch.mean(pde_residual**2)

        loss_boundary = torch.mean(pde[self.boundary_mask])**2
        #regularization
        loss_reg = self.reg_param * sum(param.abs().sum() for param in x.values())

        loss = loss_pde+loss_boundary+loss_reg
        #print(f" losst:{loss}")
        #loss = loss_boundary
        return loss

# Training Loop
def train_model(model,optimizer,obj):

    def closure():
        optimizer.zero_grad()
        loss = obj.loss(x)
        loss.backward()
        return loss


    # model.train()
    # x = list(model.parameters())
    x = model.state_dict(keep_vars=True)
    for epoch in range(epochs):
        total_loss = 0


        data_num = obj.nsamps*(obj.mshsize+1)*(obj.mshsize+1)

        # Backward pass and optimization
        optimizer.step(closure)

        total_loss += obj.loss(x).item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss/data_num:.4f}")

def plot_comparison(model, PDEObj, n_samples=1):
    # Generate data for a single sample
    # inputs_domain, inputs_boundary, inputs = PDEObj.input_data(meshsize)
    meshsize = PDEObj.mshsize
    real_solution = PDEObj.u_solution_tensor
    # real_solution = PDEObj.generate_fenics_data(n_samples=n_samples, meshsize=meshsize)[3]
    # Pass inputs_domain through the trained model to get predictions
    predicted_solution = model(PDEObj.inputs).detach().numpy()
    # Convert solutions into 2D grids
    # real_solution_grid = real_solution.squeeze(0).reshape(meshsize + 1, meshsize + 1).numpy()
    real_solution_grid = real_solution[0].reshape(meshsize + 1, meshsize + 1).numpy()
    predicted_solution_grid = predicted_solution[PDEObj.domain_mask].reshape(meshsize - 1, meshsize - 1) #not correct?
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

import copy

def loss_hessian(PDEObj, reg_param = 0.01):
    x = PDEObj.model.state_dict()
    v = x.copy() #just to get something to multiply against
    #inputs = PDEObj.input_data()[2]
    for key, vals in v.items():
       print(key)
       v[key] = vals.copy_(torch.randn(vals.size()))

    valfunc = lambda t: PDEObj.loss(t)

    torch_gradient = torch.func.grad(valfunc)
    # print(torch_gradient(x))
    def forwardoverrev(input, x, v):
        return torch.func.jvp(input, (x,), (v,))

    def hessVec(v, x):
        _, ans = forwardoverrev(torch_gradient, x, v)
        return ans

    return hessVec(v, x)

# Main Script
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = 2
    output_dim = 1
    hidden_dim = 400
    epochs = 1000
    # model = FullyConnectedNN(input_dim, hidden_dim, output_dim)
    #Loss function
    #PDE residual, for the pde residual part, we need the laplacian_nn and also the gradient of nn

    # Initialize the model, loss, and optimizer
    model     = FullyConnectedNN(input_dim, hidden_dim, output_dim).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # optimizer = optim.LBFGS(model.parameters(), lr=0.01, history_size=10, max_iter=4, line_search_fn="strong_wolfe")

    PDEObj    = PDEObjective(n_samples=1,meshsize=31,reg_param=0.0)
    PDEObj.model = model
    # loss_hessian(PDEObj)


    #print('hessvec', loss_hessian(PDEObj))


    # Train the model
    train_model(model,optimizer, PDEObj)
    # Save the model
    torch.save(model.state_dict(), "fully_connected_nn.pth")
    print("Model training complete and saved!")
    # Example usage after training the model
    plot_comparison(model, PDEObj, n_samples = 1)
