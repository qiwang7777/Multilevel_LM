from fenics import *
import numpy as np
import torch
from dolfin import *


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
    #inputs_domain = []
    #outputs_domain = []
    #inputs_boundary = []
    #outputs_boundary = []
    kappa_value_list = []
    f_value_list = []
    kappa_grad_list = []
    u_solution_list = []

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
        # Define the gradient as a vector function
        kappa_func = project(kappa,V)
        vector_space = VectorFunctionSpace(mesh, "CG", 1)
        grad_kappa = project(as_vector([kappa_func.dx(0), kappa_func.dx(1)]), vector_space)
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
        
        kappa_x_func = project(kappa_x,V)
        kappa_y_func = project(kappa_y,V)
        vector_space = VectorFunctionSpace(mesh,"CG",1)
        kappa_grad = project(as_vector([kappa_x_func,kappa_y_func]),vector_space)

        
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
        #grad_kappa_values = grad_kappa.compute_vertex_values(mesh).reshape(-1, 2)
        grad_kappa_values = kappa_grad.compute_vertex_values(mesh).reshape(-1, 2)
      
        
        f_values = f.compute_vertex_values(mesh)
        kappa_domain = kappa_values[domain_mask]
        f_domain = f_values[domain_mask]
        grad_kappa_domain = grad_kappa_values[domain_mask,:]
        kappa_value_list.append(kappa_domain)
        f_value_list.append(f_domain)
        kappa_grad_list.append(grad_kappa_domain)
        u_solution_list.append(u_array)
        
        #outputs_domain.append(u_array[domain_mask])
        #outputs_boundary.append(u_array[boundary_mask])

    # Separate domain and boundary data (only consider spatial data)
    #input_domain_features = np.stack([x_coords[domain_mask],y_coords[domain_mask]], axis=-1)
    #inputs_domain.append(input_domain_features)
    
        
        

    #input_boundary_features = np.stack([x_coords[boundary_mask],y_coords[boundary_mask]], axis=-1)
    #inputs_boundary.append(input_boundary_features)
        
    
        
        
        

    # Convert to tensors
    
    #inputs_domain = torch.tensor(np.array(inputs_domain), dtype=torch.float32).squeeze(0)
    #outputs_domain = torch.tensor(np.array(outputs_domain), dtype=torch.float32).unsqueeze(-1)
    #inputs_boundary = torch.tensor(np.array(inputs_boundary), dtype=torch.float32).squeeze(0)
    #outputs_boundary = torch.tensor(np.array(outputs_boundary), dtype=torch.float32).unsqueeze(-1)
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
def input_data(meshsize=31):
    mesh = UnitSquareMesh(meshsize, meshsize)
    vertex_coords = mesh.coordinates()
    x_coords = vertex_coords[:,0]
    y_coords = vertex_coords[:,1]
    boundary_mask = (np.isclose(x_coords, 0) | np.isclose(x_coords, 1) | 
                     np.isclose(y_coords, 0) | np.isclose(y_coords, 1))
    domain_mask = ~boundary_mask
    inputs = []
    inputs_features = np.stack([x_coords,y_coords], axis=-1)
    inputs.append(inputs_features)
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32).squeeze(0)
    inputs_domain = []
    input_domain_features = np.stack([x_coords[domain_mask],y_coords[domain_mask]], axis=-1)
    inputs_domain.append(input_domain_features)
    inputs_domain = torch.tensor(np.array(inputs_domain), dtype=torch.float32).squeeze(0)
    inputs_boundary = []
    input_boundary_features = np.stack([x_coords[boundary_mask],y_coords[boundary_mask]], axis=-1)
    inputs_boundary.append(input_boundary_features)
    inputs_boundary = torch.tensor(np.array(inputs_boundary), dtype=torch.float32).squeeze(0)
    return inputs_domain,inputs_boundary,inputs
