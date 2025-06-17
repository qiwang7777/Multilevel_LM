#Reduced Objective Function
import os
import sys
sys.path.append(os.path.abspath('..'))
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
# Fully connected neural network construction
# Set up BB
import ufl, dolfinx
from mpi4py import MPI
#import torch
import dolfinx.fem.petsc
import dolfinx.nls.petsc

import time, torch
from non_smooth.TorchVector import TorchVect
from non_smooth.L2Vectors import L2TVPrimal, L2TVDual
from non_smooth.checks import deriv_check, deriv_check_simopt, vector_check
from non_smooth.setDefaultParameters import set_default_parameters
from non_smooth.Problem import Problem
from non_smooth.trustregion import trustregion
from non_smooth.L1norm import L1TorchNorm

# Restriction operator
def restriction_R(m, n, x):
    matrix_R = x.clone()
    matrix_R.isRop = True
    I = list(matrix_R.td.items())
    J = len(I)
    for j, (k, _) in enumerate(I):
      qm  = int(np.sqrt(m[j]))
      qn  = int(np.sqrt(n[j]))
      # print(j, k, m[j], n[j], qm, qn, x.td[k].size())
      if qm == qn:
        R  = torch.eye(m[j], n[j], dtype=torch.float64)
      else:
        T  = torch.zeros((qm, qn), dtype=torch.float64) #maps fine to course
        for i in range(qm):
          if not (qn % 2 != 0 and i == qm-1):
            T[i,2*i+1] = 1/np.sqrt(2)
            T[i,2*i]   = 1/np.sqrt(2)
          else:
            T[i,2*i]   = 1
        if j == J - 1:
          R = torch.kron(T, T)
        else:
          R = torch.kron(T, T).T
      matrix_R.td[k] = R

      # print(j, k, matrix_R.td[k].size())

    return matrix_R

#Recursive step
def Reye(x):
    if x is np.ndarray:
      matrix_R = np.eye(x.shape[0])
    else:
      matrix_R       = x.clone()
      matrix_R.isRop = True
      for k, v in matrix_R.td.items():
        n = x.td[k].size()[0]
        if len(x.td[k].size()) == 1:
          matrix_R.td[k] = torch.eye(n, n, dtype=torch.float64)

        else:
          m = x.td[k].size()[1]
          matrix_R.td[k] = torch.eye(m, m, dtype=torch.float64)

    return matrix_R


class FullyConnectedNN(nn.Module):
    def __init__(self, dims):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[2], dims[3])
        self.fc3 = nn.Linear(dims[4], dims[5])
        self.activation = nn.Sigmoid() #changable

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class NNObjective:
    def __init__(self, var):
        self.var = var
    def update(self, x, type):
        return None
    # Compute objective function value
    def value_torch(self, x, ftol):
        if isinstance(x,  TorchVect):
            nn = torch.func.functional_call(self.var['NN'], x.td, (self.var['mesh'],))
        else:
          #x is the parameters of model
          nn = torch.func.functional_call(self.var['NN'], x, (self.var['mesh'],))
        f_domain = self.var['b']
        pde = torch.matmul(self.var['A'], nn.T)
        # pde_residual = pde[self.domain_mask] - f_domain
        pde_residual = pde - f_domain.T
        loss_pde = 0.5*torch.matmul(torch.matmul(self.var['M'], pde_residual).T, pde_residual)

        val = loss_pde.squeeze() # reduce dims for some reason
        #loss = loss_boundary
        return val

    def value(self, x, ftol):
        val = self.value_torch(x, ftol)
        ferr = 0.
        return val, ferr

    # Compute objective function gradient (w.r.t. u)
    def gradient(self, x, gtol):
        if isinstance(x,  TorchVect):
          grad = self.torch_gradient(x.td, gtol)
        else:
          grad = self.torch_gradient(x, gtol)
        gerr = 0
        return TorchVect(grad), gerr

    def torch_gradient(self, x, gtol):
        valfunc = lambda t: self.value_torch(t, gtol)
        g = torch.func.grad(valfunc)
        grad = g(x)
        return grad

    def hessVec(self, v, x, htol):

        gfunc = lambda t: self.torch_gradient(t, htol)
        _, ans = torch.func.jvp(gfunc, (x.td,), (v.td,))
        return TorchVect(ans), 0

class NNSetup:
    """
    Solve the NN control problem
    """
    def __init__(self, NN_dim, n, alpha, beta, n_samples = 1):
        self.n      = n
        self.NN_dim = NN_dim
        self.nsamps = n_samples
        # Mesh grid for x, y ((meshsize+1) x (meshsize+1))
        self.domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n,n)

        self.V      = dolfinx.fem.functionspace(self.domain, ("Lagrange", 1))
        # Source term
        self.f      = dolfinx.fem.Constant(self.domain, dolfinx.default_scalar_type(-6))
        self.inputs = torch.tensor(self.domain.geometry.x[:,0:1], dtype=torch.float64).T

        # Boundary condition
        self.u_D = dolfinx.fem.Function(self.V)
        self.u_D.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
        tdim            = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim - 1, tdim)
        boundary_facets = dolfinx.mesh.exterior_facet_indices(self.domain.topology)
        boundary_dofs   = dolfinx.fem.locate_dofs_topological(self.V, self.domain.topology.dim - 1, boundary_facets)
        self.bc         = dolfinx.fem.dirichletbc(self.u_D, boundary_dofs)

        self.beta  = beta
        self.alpha = alpha
        self.R     = []



        self.NN    = FullyConnectedNN(NN_dim)
        self.NN.to(torch.float64)

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
            x     = ufl.SpatialCoordinate(self.domain)
            K     = dolfinx.fem.Function(self.V)
            K.interpolate(lambda x:  1.1 + np.cos(kx*np.pi*(np.cos(alpha)*(x[0]-0.5) - np.sin(alpha)*(x[1]-0.5) + 0.5 + ax)) * np.cos(ky*np.pi*(np.sin(alpha)*(x[0]-0.5) + np.cos(alpha)*(x[1]-0.5) + 0.5 + ay)))

            V = self.V
            # Define variational problem
            v = ufl.TestFunction(V)
            uh = dolfinx.fem.Function(V)
            u  = ufl.TrialFunction(V)
            F = ufl.dot(K*ufl.grad(uh), ufl.grad(v)) * ufl.dx  - ufl.inner(self.f, v) * ufl.dx
            p = dolfinx.fem.petsc.NonlinearProblem(F, uh, bcs = [self.bc])
            solver = dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, p)
            solver.solve(uh)

            #get operators
            a = ufl.dot(K*ufl.grad(u), ufl.grad(v)) * ufl.dx
            l = self.f * v * ufl.dx
            cost =  1 / 2 * (uh ) * (uh) * ufl.dx #+ alpha / 2 * f**2 * ufl.dx
            ac = dolfinx.fem.form(a)
            H = dolfinx.fem.form(ufl.derivative(ufl.derivative(cost, uh), uh))
            A = dolfinx.fem.create_matrix(ac)
            M = dolfinx.fem.create_matrix(H)
            b = dolfinx.fem.Function(uh.function_space)
            L = dolfinx.fem.form(l)
            # Solve the problem
            # Get the solution and input features
            dolfinx.fem.assemble_matrix(A, ac, bcs = [self.bc])
            dolfinx.fem.assemble_matrix(M, H)
            AT = torch.from_numpy(A.to_dense()) #stiffness
            MT = torch.from_numpy(M.to_dense()) #mass
            dolfinx.fem.assemble_vector(b.x.array, L) #rhs
            dolfinx.fem.apply_lifting(b.x.array, [ac], [[self.bc]])
            b.x.scatter_reverse(dolfinx.la.InsertMode.add)
            [bc.set(b.x.array) for bc in [self.bc]]

            ut = dolfinx.fem.Function(V)
            ut.x.array[:] = np.linalg.solve(AT, b.x.array)
            self.A            = AT.to(dtype=torch.float64)
            self.M            = MT.to(dtype=torch.float64)
            u_array           = uh.x.array[:]

            kappa_values      = [] #K.x.array[:]
            f_values     = b.x.array[:] #self.f.x.array[:]
            # print(np.dot(A, u_array) - f_values)

            kappa_domain      = kappa_values #[self.domain_mask]
            f_domain          = f_values #[self.domain_mask]

            kappa_value_list.append(kappa_domain)
            f_value_list.append(f_domain)
            u_solution_list.append(u_array)


        kappa_value_list  = torch.tensor(np.array(kappa_value_list), dtype=torch.float64)
        f_value_list      = torch.tensor(np.array(f_value_list), dtype=torch.float64)
        u_solution_tensor = torch.tensor(np.array(u_solution_list),dtype=torch.float64)


        return  kappa_value_list, f_value_list, u_solution_tensor
    def returnVars(self, useEuclidean):
      var = {'beta':self.beta,
             'n':self.n,
             'nu':self.n,
             'nz':sum(p.numel() for p in self.NN.parameters()),
             'alpha':self.alpha,
             'A':self.A,
             'M':self.M,
             'NN':self.NN,
             'k':self.kappa_value_list,
             'b':self.f_value_list,
             'ud':self.u_solution_tensor,
             'useEuclidean':useEuclidean,
             'mesh':torch.tensor(self.domain.geometry.x[:,0:1], dtype=torch.float64).T
            }
      return var

    def plot_solutions(self, pinns_solution=None):

        # Extract mesh coordinates
        geometry = self.domain.geometry.x[:, :2]
        #X = geometry[:, 0]
        #Y = geometry[:, 1]


        # Real solution
        real_solution = self.u_solution_tensor[0].numpy()  # Shape: (1024,)

        # Predict the PINNs solution if not provided
        if pinns_solution is None:
            with torch.no_grad():
                pinns_solution = self.NN(self.inputs).numpy()  # Shape: (1, 1024)
            pinns_solution = pinns_solution.squeeze(0)  # Remove the extra dimension -> (1024,)


        # Plot the true solution
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("True Solution")
        plt.tricontourf(geometry[:, 0], geometry[:, 1], real_solution, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

        # Plot the predicted solution
        plt.subplot(1, 3, 2)
        plt.title("Predicted Solution")
        plt.tricontourf(geometry[:, 0], geometry[:, 1], pinns_solution, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

        # Plot the error (absolute difference)
        error = np.abs(real_solution - pinns_solution)
        plt.subplot(1, 3, 3)
        plt.title("Error (Absolute Difference)")
        plt.tricontourf(geometry[:, 0], geometry[:, 1], error, levels=50, cmap='viridis')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")

        plt.tight_layout()
        plt.show()


def driver(savestats, name):
    print("driver started")
    np.random.seed(0)

    # Set up optimization problem
    n          = [30, 30] #[30, 15]# Number of cells
    NN_dim     = np.array([(n[0]+1)**2, 100, 100, 100, 100, (n[0]+1)**2]) # Neural network nodes
    meshlist   = [NN_dim, np.array([(n[1]+1)**2, 100, 100, 100, 100, (n[1]+1)**2])]
    # meshlist   = [NN_dim, np.array([(n[0]+1)**2, 49, 49, 25, 25, (n[0]+1)**2])]


    alpha      = 1  # L2 penalty parameter
    beta       = 1e-4  # L1 penalty parameter
    derivCheck = False


    problems = [] #problem list goes from fine to coarse
    x0       = []
    for i in range(0, len(meshlist)):
        nnset = NNSetup(meshlist[i], n[i], alpha, beta, n_samples=1)
        x     = TorchVect(nnset.NN.state_dict())
        if i == 0:
           x0 = x
        var   = nnset.returnVars(False)
        if i < len(meshlist) - 1:
          R = restriction_R(meshlist[i+1], meshlist[i], x) #puts R in preceeding problem
        else:
          R = Reye(x)
        p = Problem(var, R)
        p.obj_smooth    = NNObjective(var)
        p.obj_nonsmooth = L1TorchNorm(var)
        p.pvector = L2TVPrimal(var)
        p.dvector = L2TVDual(var)
        problems.append(p)

        if derivCheck:
            for i in range(0, len(meshlist)):
                x =  TorchVect(nnset.NN.state_dict())
                d = x.clone() #just to get something to multiply against
                for key, vals in d.td.items():
                    d.td[key] = vals.copy_(torch.randn(vals.size()))
                    deriv_check(x, d, problems[i], 1e-4 * np.sqrt(np.finfo(float).eps))
                    vector_check(x, d, problems[i])

    cnt = {}
    # Update default parameters
    params = set_default_parameters("SPG2")
    params["reltol"] = False
    params["t"] = 2 / alpha
    params["ocScale"] = 1 / alpha
    params['maxit'] = 300

    #Plot loss
    #loss_values = []

    # Solve optimization problem
    start_time   = time.time()
    x, cnt_tr    = trustregion(0, x0, params['delta'],problems, params)
    loss_history = cnt_tr['objhist']
    #loss_history_single = cnt_single['objhist']
    iterations = range(1, len(loss_history) + 1)
    #iterations_single = range(1, len(loss_history_single) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(iterations, loss_history, marker="o", color="b")
    #plt.plot(iterations_single, loss_history_single, marker="o", color="r")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Objective Function)")
    plt.title("Loss vs Iterations in Trust-Region Optimization")
    plt.grid(True)
    plt.show()



    elapsed_time = time.time() - start_time

    print(f"Optimization completed in {elapsed_time:.2f} seconds")

    pro_tr =  []


    cnt[1] = (cnt_tr, pro_tr)


    print("\nSummary")
    print(
        "           niter     nobjs     ngrad     nhess     nobjn     nprox     ")


    print(
        f"   SGP2:  {cnt[1][0]['iter']:6d}    {cnt[1][0]['nobj1']:6d}    {cnt[1][0]['ngrad']:6d}    {cnt[1][0]['nhess']:6d}    "
        f"{cnt[1][0]['nobj2']:6d}    {cnt[1][0]['nprox']:6d}     "
    )

    ## Print updated weights of the first layer (fc1)
    #nnset.NN.load_state_dict(x.td)  # Load the updated parameters into the neural network
    #nnset.NN.eval()
    #state_dict_after  = nnset.NN.state_dict()
    #weights_fc1_after = state_dict_after['fc1.weight']
    #print("Weights of fc1 after optimization:", torch.nonzero(weights_fc1_after))

    final_nnset = NNSetup(meshlist[0], n[0], alpha, beta, n_samples=1)
    final_nnset.NN.load_state_dict(x.td)
    final_nnset.NN.eval()



    print("Updated neural network is stored in `updated_nn`.")
    final_nnset.plot_solutions()


    return cnt


cnt = driver(False, "test_run")
