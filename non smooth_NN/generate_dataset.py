#In this file, we generate the dataset, input dimension is 2 (spatial data in 2d) and output dimension is 1
#To get all the values we need to put into loss function, including the value of kappa and f.
from fenics import *
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch

def generate_fenics_data(n_samples=100, meshsize = 31):
    # Generate random values for kx, ky, ax, ay, and alpha
    kx_samples = np.random.uniform(0.5, 4.0, n_samples)
    ky_samples = np.random.uniform(0.5, 4.0, n_samples)
    ax_samples = np.random.uniform(0.0, 0.5, n_samples)
    ay_samples = np.random.uniform(0.0, 0.5, n_samples)
    alpha_samples = np.random.uniform(0.0, np.pi / 2, n_samples)

    # Mesh grid for x, y ((meshsize+1) x (meshsize+1))
    mesh = UnitSquareMesh(meshsize, meshsize)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Boundary condition
    u_D = Constant(0.0)
    bc = DirichletBC(V, u_D, "on_boundary")

    # Prepare containers for inputs and outputs
    inputs_domain = []
    outputs_domain = []
    inputs_boundary = []
    outputs_boundary = []
    kappa_value_list = []
    f_value_list = []

    # Create vertex coordinates for mapping
    vertex_coords = mesh.coordinates()
    x_coords = vertex_coords[:, 0]
    
    y_coords = vertex_coords[:, 1]

    # Define masks for boundary and domain
    boundary_mask = (np.isclose(x_coords, 0) | np.isclose(x_coords, 1) | 
                     np.isclose(y_coords, 0) | np.isclose(y_coords, 1))
    domain_mask = ~boundary_mask

    for i in range(n_samples):
        kx = kx_samples[i]
        ky = ky_samples[i]
        ax = ax_samples[i]
        ay = ay_samples[i]
        alpha = alpha_samples[i]
        

        # Define rotated coordinates x' and y'
        kappa = Expression(
            "1.1 + cos(kx*pi*(cos(alpha)*(x[0]-0.5) - sin(alpha)*(x[1]-0.5) + 0.5 + ax)) * "
            "cos(ky*pi*(sin(alpha)*(x[0]-0.5) + cos(alpha)*(x[1]-0.5) + 0.5 + ay))",
            degree=2, kx=kx, ky=ky, ax=ax, ay=ay, alpha=alpha, pi=np.pi
        )
        
        # Source term
        f = Expression("32*exp(-4*((x[0]-0.25)*(x[0]-0.25) + (x[1]-0.25)*(x[1]-0.25)))", degree=2)
        
        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        a = dot(kappa * grad(u), grad(v)) * dx
        L = f * v * dx
        
        # Solve the problem
        u_sol = Function(V)
        solve(a == L, u_sol, bc)

        # Get the solution and input features
        u_array = u_sol.compute_vertex_values(mesh)
        kappa_values = kappa.compute_vertex_values(mesh)
      
        
        f_values = f.compute_vertex_values(mesh)
        kappa_domain = kappa_values[domain_mask]
        f_domain = f_values[domain_mask]
        kappa_value_list.append(kappa_domain)
        f_value_list.append(f_domain)
        outputs_domain.append(u_array[domain_mask])
        outputs_boundary.append(u_array[boundary_mask])

    # Separate domain and boundary data (only consider spatial data)
    input_domain_features = np.stack([x_coords[domain_mask],y_coords[domain_mask]], axis=-1)
    inputs_domain.append(input_domain_features)
    
        
        

    input_boundary_features = np.stack([x_coords[boundary_mask],y_coords[boundary_mask]], axis=-1)
    inputs_boundary.append(input_boundary_features)
        
    
        
        
        

    # Convert to tensors
    
    inputs_domain = torch.tensor(np.array(inputs_domain), dtype=torch.float32)
    outputs_domain = torch.tensor(np.array(outputs_domain), dtype=torch.float32).unsqueeze(-1)
    inputs_boundary = torch.tensor(np.array(inputs_boundary), dtype=torch.float32)
    outputs_boundary = torch.tensor(np.array(outputs_boundary), dtype=torch.float32).unsqueeze(-1)
    kappa_value_list = torch.tensor(np.array(kappa_value_list), dtype=torch.float32)
    f_value_list= torch.tensor(np.array(f_value_list), dtype=torch.float32)
    

    return inputs_domain, outputs_domain, inputs_boundary, outputs_boundary, kappa_value_list, f_value_list
