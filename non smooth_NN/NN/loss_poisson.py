import torch
import torch.nn as nn

class PDELoss(nn.Module):
    def __init__(self, kappa, f, domain_points, boundary_points, lambda_reg=0.001):
        super(PDELoss, self).__init__()
        self.kappa = kappa
        self.f = f
        self.domain_points = domain_points
        self.boundary_points = boundary_points
        self.lambda_reg = lambda_reg

    def forward(self, model):
        # Residual Loss
        domain_points = self.domain_points.requires_grad_(True)
        u_pred = model(domain_points)
        grad_u = torch.autograd.grad(u_pred, domain_points, grad_outputs=torch.ones_like(u_pred),
                                      create_graph=True)[0]
        div_kappa_grad_u = torch.autograd.grad(self.kappa(domain_points) * grad_u, domain_points,
                                               grad_outputs=torch.ones_like(grad_u), create_graph=True)[0]
        residual_loss = torch.mean((div_kappa_grad_u + self.f(domain_points))**2)

        # Boundary Loss
        boundary_points = self.boundary_points
        u_boundary_pred = model(boundary_points)
        boundary_loss = torch.mean(u_boundary_pred**2)  # Dirichlet boundary: u = 0
        
        #Regularization
        l1_norm = sum(torch.sum(torch.abs(param)) for param in model.parameters())

        # Total Loss
        total_loss = residual_loss + boundary_loss + self.lambda_reg*l1_norm
        return total_loss
