#In this file, we generate the dataset, input dimension is 2 (spatial data in 2d) and output dimension is 1
#To get all the values we need to put into loss function, including the value of kappa and f.
import ufl, dolfinx
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector, set_bc
import dolfinx.nls.petsc as PETSC
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import pyvista
from dolfinx import plot
pyvista.start_xvfb()


# Fully connected neural network construction
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear((input_dim+1)**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, (output_dim+1)**2)
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
        self.domain = mesh.create_unit_square(MPI.COMM_WORLD, meshsize, meshsize)
        self.V = fem.functionspace(self.domain, ("Lagrange", 1))
        # Source term
        self.f = fem.Constant(self.domain, dolfinx.default_scalar_type(-6))
        self.inputs = torch.tensor(self.domain.geometry.x[:,0:1], dtype=torch.float32).T

        # Boundary condition
        self.u_D = fem.Function(self.V)
        self.u_D.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
        tdim            = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim - 1, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        boundary_dofs   = fem.locate_dofs_topological(self.V, self.domain.topology.dim - 1, boundary_facets)
        self.bc         = fem.dirichletbc(self.u_D, boundary_dofs)


        t = self.generate_fenics_data()
        self.kappa_value_list  = t[0]
        self.f_value_list      = t[1]
        # self.kappa_grad_list   = t[2]
        self.u_solution_tensor = t[2]

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
            # kappa = Expression(
            #     "1.1 + cos(kx*pi*(cos(alpha)*(x[0]-0.5) - sin(alpha)*(x[1]-0.5) + 0.5 + ax)) * cos(ky*pi*(sin(alpha)*(x[0]-0.5) + cos(alpha)*(x[1]-0.5) + 0.5 + ay))",
            #     degree=2, kx=kx, ky=ky, ax=ax, ay=ay, alpha=alpha, pi=np.pi
            # )
            # self.kappa = kappa
            x     = ufl.SpatialCoordinate(self.domain)
            K     = fem.Function(self.V)
            K.interpolate(lambda x:  1.1 + np.cos(kx*np.pi*(np.cos(alpha)*(x[0]-0.5) - np.sin(alpha)*(x[1]-0.5) + 0.5 + ax)) * np.cos(ky*np.pi*(np.sin(alpha)*(x[0]-0.5) + np.cos(alpha)*(x[1]-0.5) + 0.5 + ay)))

            # Kappa =  1.1 + np.cos(kx*np.pi*(np.cos(alpha)*(x[0]-0.5) - np.sin(alpha)*(x[1]-0.5) + 0.5 + ax)) * np.cos(ky*np.pi*(np.sin(alpha)*(x[0]-0.5) + np.cos(alpha)*(x[1]-0.5) + 0.5 + ay))
            # K.interpolate(fem.Expression(Kappa, V.element.interpolation_points()))

            V = self.V
            # Define variational problem
            v = ufl.TestFunction(V)
            # a = dot(kappa * grad(u), grad(v)) * dx
            uh = fem.Function(V)
            u  = ufl.TrialFunction(V)
            F = ufl.dot(K*ufl.grad(uh), ufl.grad(v)) * ufl.dx  - ufl.inner(self.f, v) * ufl.dx
            # L = self.f * v * ufl.dx
            p = fem.petsc.NonlinearProblem(F, uh, bcs = [self.bc])
            solver = PETSC.NewtonSolver(MPI.COMM_WORLD, p)
            solver.solve(uh)
            #get operators
            a = ufl.dot(K*ufl.grad(u), ufl.grad(v)) * ufl.dx
            l = self.f * v * ufl.dx
            cost =  1 / 2 * (uh ) * (uh) * ufl.dx #+ alpha / 2 * f**2 * ufl.dx
            ac = fem.form(a)
            H = dolfinx.fem.form(ufl.derivative(ufl.derivative(cost, uh), uh))
            # ac = fem.form(ufl.derivative(F, uh))
            A = fem.create_matrix(ac)
            M = fem.create_matrix(H)
            b = dolfinx.fem.Function(uh.function_space)
            L = dolfinx.fem.form(l)
            # Solve the problem
            # Get the solution and input features
            fem.assemble_matrix(A, ac, bcs = [self.bc])
            fem.assemble_matrix(M, H)
            AT = torch.from_numpy(A.to_dense()) #stiffness
            MT = torch.from_numpy(M.to_dense()) #mass
            fem.assemble_vector(b.x.array, L) #rhs
            fem.apply_lifting(b.x.array, [ac], [[self.bc]])
            b.x.scatter_reverse(dolfinx.la.InsertMode.add)
            [bc.set(b.x.array) for bc in [self.bc]]

            ut = dolfinx.fem.Function(V)
            ut.x.array[:] = np.linalg.solve(AT, b.x.array)
            self.A            = AT.to(dtype=torch.float32)
            self.M            = MT.to(dtype=torch.float32)
            u_array           = uh.x.array[:]
            print('Test A \\ b = u vs fenicsx solution: ', np.linalg.norm(u_array - ut.x.array[:]))
            print('Test ||A*u - f||^2', np.linalg.norm(np.dot(self.A, ut.x.array[:]) - b.x.array[:]))
            V2 = fem.functionspace(self.domain, ("Lagrange", 2))
            uex = fem.Function(V2)
            uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
            L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
            error_local = fem.assemble_scalar(L2_error)
            error_L2 = np.sqrt(self.domain.comm.allreduce(error_local, op=MPI.SUM))
            print('Error l2: ', error_L2)
            kappa_values      = [] #K.x.array[:]
            f_values          = b.x.array[:] #self.f.x.array[:]
            # print(np.dot(A, u_array) - f_values)

            kappa_domain      = kappa_values #[self.domain_mask]
            f_domain          = f_values #[self.domain_mask]

            kappa_value_list.append(kappa_domain)
            f_value_list.append(f_domain)
            u_solution_list.append(u_array)


        kappa_value_list  = torch.tensor(np.array(kappa_value_list), dtype=torch.float32)
        f_value_list      = torch.tensor(np.array(f_value_list), dtype=torch.float32)
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
        nn = torch.func.functional_call(self.model, x, (self.inputs,))
        f_domain = self.f_value_list
        pde = torch.matmul(self.A, nn.T)
        #I suppose we can also domain_mask A?
        # pde_residual = pde[self.domain_mask] - f_domain
        pde_residual = pde - f_domain.T
        loss_pde = 0.5*torch.matmul(torch.matmul(self.M, pde_residual).T, pde_residual)
        # loss_boundary = torch.mean(pde[self.boundary_mask])**2
        #regularization
        # loss_reg = self.reg_param * sum(param.abs().sum() for param in x.values())

        # loss = loss_pde+loss_boundary+loss_reg
        loss = loss_pde
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
        if epoch % 100 == 0:
          print(f"Epoch [{epoch}/{epochs}], Loss: {total_loss:.8f}")

def plot_comparison(model, PDEObj, n_samples=1):
    # Generate data for a single sample
    # inputs_domain, inputs_boundary, inputs = PDEObj.input_data(meshsize)
    u_topology, u_cell_types, u_geometry = plot.vtk_mesh(PDEObj.V)
    u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
    u_grid.point_data["u"] = PDEObj.u_solution_tensor[0]
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        # u_plotter.show()
        u_plotter.save_graphic("true_state.eps")
    # Pass inputs_domain through the trained model to get predictions
    predicted_solution = model(PDEObj.inputs).detach().numpy()
    u_grid.point_data["u"] = predicted_solution[0]
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        # u_plotter.show()
        u_plotter.save_graphic("nn.eps")
    # Compute error (absolute difference)
    error_grid = abs(PDEObj.u_solution_tensor - predicted_solution)
    print(np.linalg.norm(error_grid), np.linalg.norm(np.dot(PDEObj.A, predicted_solution.T) - PDEObj.f_value_list.detach().numpy()))
    u_grid.point_data["u"] = error_grid[0]
    u_grid.set_active_scalars("u")
    u_plotter = pyvista.Plotter()
    u_plotter.add_mesh(u_grid, show_edges=True)
    u_plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        # u_plotter.show()
        u_plotter.save_graphic("diff.eps")


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
    ms        = 31
    input_dim = ms
    output_dim = ms
    hidden_dim = 100
    epochs = 1000
    # model = FullyConnectedNN(input_dim, hidden_dim, output_dim)
    #Loss function
    #PDE residual, for the pde residual part, we need the laplacian_nn and also the gradient of nn

    # Initialize the model, loss, and optimizer
    model     = FullyConnectedNN(input_dim, hidden_dim, output_dim).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.LBFGS(model.parameters(), lr=0.01, history_size=10, max_iter=4, line_search_fn="strong_wolfe")

    PDEObj    = PDEObjective(n_samples=1,meshsize=ms,reg_param=0.0)
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
