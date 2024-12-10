import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from fenics import *

# Parameters for neural network and FEniCS setup
input_dim = 3 * 32 * 32  # Flattened input size (3 channels, 32x32 grid)
output_dim = 32 * 32     # Flattened output size (32x32 grid)
hidden_neurons = 400     # Neurons in hidden layers
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
        
        x = x.view(-1, 32, 32)
        return x

# Generate data using FEniCS
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
    inputs = []
    outputs = []

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

        # Get the solution as a numpy array
        u_array = u_sol.compute_vertex_values(mesh)
        u_array = u_array.reshape(32, 32)  

        # Stack kappa, f, and mesh coordinates as input
        # Create the mesh grid coordinates
        X, Y = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32))  # Correct the grid to 32x32

        # Stack the kappa, f, X, and Y as input features
        input_features = np.stack([kappa.compute_vertex_values(mesh), f.compute_vertex_values(mesh), X.flatten(), Y.flatten()], axis=0)
        input_features = input_features[:3, :32*32].reshape(3, 32, 32)
        inputs.append(input_features)
        outputs.append(u_array)

    # Convert to tensors
    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.array(outputs), dtype=torch.float32)

    return inputs, outputs

# Load Dataset
def load_data(n_samples=10000):
    inputs, outputs = generate_fenics_data(n_samples)
    dataset = TensorDataset(inputs, outputs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training Loop
def train_model(model, dataloader, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            targets = targets.view(-1, 32, 32) 
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# Main Script
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the model, loss, and optimizer
    model = FullyConnectedNN(input_dim, hidden_neurons, output_dim).to(device)
    criterion = nn.MSELoss()  # Mean squared error for regression
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load data
    dataloader = load_data()

    # Train the model
    train_model(model, dataloader, criterion, optimizer)

    # Save the model
    torch.save(model.state_dict(), "fully_connected_nn.pth")
    print("Model training complete and saved!")


#print(generate_fenics_data(10000)[0].shape)
#print(generate_fenics_data(10000)[1].shape)       
import matplotlib.pyplot as plt

# Function to plot the exact solution and NN learned solution
def plot_comparison(model, dataloader, index=0):
    model.eval()
    
    # Get one batch of data
    inputs, targets = next(iter(dataloader))
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Get the model's prediction for the input at the given index
    prediction = model(inputs).cpu().detach().numpy()
    exact_solution = targets.cpu().detach().numpy()
    
    # Select a specific sample from the batch for visualization
    exact_u = exact_solution[index].reshape(32, 32)
    predicted_u = prediction[index].reshape(32, 32)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Exact solution
    im1 = axes[0].imshow(exact_u, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    axes[0].set_title("Exact Solution")
    fig.colorbar(im1, ax=axes[0])
    
    # NN predicted solution
    im2 = axes[1].imshow(predicted_u, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    axes[1].set_title("NN Predicted Solution")
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_comparison(model, dataloader, index=0)
      
    
    
    

