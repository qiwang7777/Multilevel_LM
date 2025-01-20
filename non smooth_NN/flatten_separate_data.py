import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fenics import *

# Parameters for neural network and FEniCS setup
input_dim = 4096  # Flattened input size (4 channels (kappa,f,x,y), 1024 grid points)
output_dim = 1024     # Flattened output size (flattened solution values u)
hidden_dim = 400     # Neurons in hidden layers
batch_size = 64          # Batch size for training
epochs = 50              # Number of training epochs
learning_rate = 0.001    # Learning rate

# Neural Network Architecture (Fully Connected NN)
class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        
        
        return x
    
    
    

def generate_fenics_data(n_samples=10000):
    # Generate random values for kx, ky, ax, ay, and alpha
    kx_samples = np.random.uniform(0.5, 4.0, n_samples)
    ky_samples = np.random.uniform(0.5, 4.0, n_samples)
    ax_samples = np.random.uniform(0.0, 0.5, n_samples)
    ay_samples = np.random.uniform(0.0, 0.5, n_samples)
    alpha_samples = np.random.uniform(0.0, np.pi / 2, n_samples)

    # Mesh grid for x, y (32x32)
    mesh = UnitSquareMesh(31, 31)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Boundary condition
    u_D = Constant(0.0)
    bc = DirichletBC(V, u_D, "on_boundary")

    # Prepare containers for inputs and outputs
    inputs_domain = []
    outputs_domain = []
    inputs_boundary = []
    outputs_boundary = []

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

        # Separate domain and boundary data
        input_domain_features = np.stack([
            kappa_values[domain_mask],
            f_values[domain_mask],
            x_coords[domain_mask],
            y_coords[domain_mask]
        ], axis=-1).reshape(-1,4)
        inputs_domain.append(input_domain_features.flatten())
        outputs_domain.append(u_array[domain_mask].flatten())

        input_boundary_features = np.stack([
            kappa_values[boundary_mask],
            f_values[boundary_mask],
            x_coords[boundary_mask],
            y_coords[boundary_mask]
        ], axis=-1).reshape(-1,4)
        inputs_boundary.append(input_boundary_features.flatten())
        outputs_boundary.append(u_array[boundary_mask].flatten())

    # Convert to tensors
    inputs_domain = torch.tensor(np.array(inputs_domain), dtype=torch.float32)
    outputs_domain = torch.tensor(np.array(outputs_domain), dtype=torch.float32)
    inputs_boundary = torch.tensor(np.array(inputs_boundary), dtype=torch.float32)
    outputs_boundary = torch.tensor(np.array(outputs_boundary), dtype=torch.float32)

    return inputs_domain, outputs_domain, inputs_boundary, outputs_boundary

def data_separate(n_samples=10000):
    inputs_domain = generate_fenics_data(n_samples)[0]
    kappa_domain = inputs_domain[:,0::4]
    f_domain = inputs_domain[:,1::4]
    x_domain = inputs_domain[:,2::4]
    y_domain = inputs_domain[:,3::4]
    inputs_boundary = generate_fenics_data(n_samples)[2]
    kappa_boundary = inputs_boundary[:,0::4]
    f_boundary = inputs_boundary[:,1::4]
    x_boundary = inputs_boundary[:,2::4]
    y_boundary = inputs_boundary[:,3::4]
    return kappa_domain, f_domain, x_domain, y_domain, kappa_boundary, f_boundary, x_boundary, y_boundary
 
def loss_pde(model,n_samples=10000,reg_param=0.01):
    kappa_domain, f_domain, x_domain, y_domain, kappa_boundary, f_boundary, x_boundary, y_boundary = data_separate(n_samples)
    x_domain = x_domain.requires_grad_()
    inputs_domain = generate_fenics_data(n_samples)[0]
    inputs_domain = inputs_domain.requires_grad_()
    inputs_boundary = generate_fenics_data(n_samples)[2]
    inputs = torch.cat((inputs_domain,inputs_boundary),dim=1)
    
    inputs = inputs.requires_grad_()
    u = model(inputs)

    
    #print(u.shape)
    u = u.requires_grad_()
    #print(inputs.requires_grad)
    #print(u.requires_grad)
    grad_u = torch.autograd.grad(outputs = u, inputs = inputs, grad_outputs = torch.ones_like(u),
                                 create_graph=True, only_inputs = True)[0]
    grad_u = grad_u.view(n_samples,1024,4)
    grad_u_x = grad_u[:,:,2]
    #print(grad_u_x.shape)
    grad_u_y = grad_u[:,:,3]
    
    grad_u_x_domain = grad_u_x[:,:900] #as the inputs = torch.cat((inputs_domain,inputs_boundary),dim=1)
    grad_u_y_domain = grad_u_y[:,:900]
    
    
    
    kappa_domain = data_separate(n_samples)[0]
    
    f_domain = data_separate(n_samples)[1]
    
    #Compute $\kappa\nabla u$
   
    kappa_grad_u_x = kappa_domain*grad_u_x_domain
    kappa_grad_u_y = kappa_domain*grad_u_y_domain
   
    #Compute divergence $\nable\cdot(\kappa\nabla u)=\kappa\Delta u+\nabla\kappa\cdot\nabla u$
    
    grad_kappa_grad_u_x = torch.autograd.grad(
        outputs=kappa_grad_u_x.sum(dim=1), inputs=inputs_domain,
        grad_outputs=torch.ones_like(kappa_grad_u_x.sum(dim=1)),
        create_graph=True, only_inputs=True)[0]
    
    grad_kappa_grad_u_x = grad_kappa_grad_u_x.view(n_samples,900,4)
    div_kappa_grad_u_x = grad_kappa_grad_u_x[:,:,2]
    
    grad_kappa_grad_u_y = torch.autograd.grad(
        outputs=kappa_grad_u_y.sum(dim=1), inputs=inputs_domain,
        grad_outputs=torch.ones_like(kappa_grad_u_y.sum(dim=1)),
        create_graph=True, only_inputs=True)[0]
    grad_kappa_grad_u_y = grad_kappa_grad_u_y.view(n_samples,900,4)
    
    div_kappa_grad_u_y = grad_kappa_grad_u_y[:,:,3]
    
    div_kappa_grad_u = div_kappa_grad_u_x+div_kappa_grad_u_y
    pde_residual = -div_kappa_grad_u-f_domain
    loss_pde = torch.mean(pde_residual**2)
    
    #boundary residual
    u_boundary = u[:,900:]
    loss_boundary = torch.mean(u_boundary**2)
    
    #reguralization
    loss_reg = reg_param * sum(param.abs().sum() for param in model.parameters())
    
    loss = loss_pde+loss_boundary+loss_reg
    
    
    return loss
   

model = FullyConnectedNN(input_dim,hidden_dim,output_dim)


# Training Loop
def train_model(model, n_samples=10000,epochs=50, batch_size=64,learning_rate=0.001,reg_param = 0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Generate data
    inputs_domain, outputs_domain, inputs_boundary, outputs_boundary = generate_fenics_data(n_samples)
    
    # Concatenate domain and boundary inputs
    inputs = torch.cat([inputs_domain, inputs_boundary], dim=1).to(device)
    outputs = torch.cat([outputs_domain, outputs_boundary], dim=1).to(device)
    
    # Create dataset and DataLoader
    dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            batch_inputs, batch_outputs = batch
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = loss_pde(model, n_samples, reg_param=reg_param)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Log epoch loss
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
    
    print("Training complete.")
