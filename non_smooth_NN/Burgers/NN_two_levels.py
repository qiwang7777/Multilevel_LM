#Reduced Objective Function
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
# Fully connected neural network construction
# Set up BB
import ufl, dolfinx
from mpi4py import MPI
import torch
import dolfinx.fem.petsc
import dolfinx.nls.petsc
from TorchVector import TorchVect
import time
from setDefaultParameters import set_default_parameters
from checks import deriv_check, vector_check

# Problem Class
class L2vectorPrimal:
    def __init__(self, var):
        self.var = var
    @torch.no_grad()
    def dot(self, x, y):
        ans = 0
        for k, v in x.td.items():
            ans += torch.sum(torch.mul(v, y.td[k]))
        return ans.item()
    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x, y)
    @torch.no_grad()
    def norm(self, x):
        return np.sqrt(self.dot(x, x))
    @torch.no_grad()
    def dual(self, x):
        return x
class L2vectorDual:
    def __init__(self, var):
        self.var = var
    @torch.no_grad()
    def dot(self, x, y):
        ans = 0
        for k, v in x.td.items():
            ans += torch.sum(torch.mul(v, y.td[k]))
        return ans.item()
    @torch.no_grad()
    def apply(self, x, y):
        return self.dot(x,y)
    @torch.no_grad()
    def norm(self, x):
        return np.sqrt(self.dot(x, x))
    @torch.no_grad()
    def dual(self, x):
        return x
class L1Norm:
    def __init__(self, var):
        self.var = var

    def value(self, x):
        val = 0
        for k, v in x.td.items():
            val += torch.sum(torch.abs(v))
        return self.var['beta'] * val

    def prox(self, x, t):
        temp = x.clone()
        for k, v in x.td.items():
            temp.td[k] = torch.max(torch.tensor([0.0]), torch.abs(v) - t*self.var['beta'])*torch.sign(v)
        # return np.maximum(0, np.abs(x) - t * self.var['Rlump'] * self.var['beta']) * np.sign(x)
        return temp

    def dir_deriv(self, s, x):
        sx = np.sign(x)
        return self.var['beta'] * (np.dot(sx.T, s) + np.dot((1 - np.abs(sx)).T, np.abs(s)))

    def project_sub_diff(self, g, x):
        sx = np.sign(x)
        return self.var['beta'] * sx + (1 - np.abs(sx)) * np.clip(g, -self.var['beta'], self.var['beta'])

    def gen_jac_prox(self, x, t):
        d = np.ones_like(x)
        px = self.prox(x, t)
        ind = px == 0
        d[ind] = 0
        return np.diag(d), ind

    def apply_prox_jacobian(self, v, x, t):
        if self.var['useEuclidean']:
            ind = np.abs(x) <= t * self.var['Rlump'] * self.var['beta']
        else:
            ind = np.abs(x) <= t * self.var['beta']
        Dv = v.copy()
        Dv[ind] = 0
        return Dv

    def get_parameter(self):
        return self.var['beta']
class Problem:
    def __init__(self, var):
        self.var = var
        self.pvector = L2vectorPrimal(var)
        self.dvector = L2vectorDual(var)
        self.obj_smooth = NNObjective(var)
        self.obj_nonsmooth = L1Norm(var)

# Restriction operator
def restriction_R(m,n):
    matrix_R = torch.zeros((m,n), dtype=torch.float64)
    for i in range(m):
        matrix_R[i,2*(i+1)-1] = 1/np.sqrt(2)
        matrix_R[i,2*i] = 1/np.sqrt(2)
    return matrix_R

class FullyConnectedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FullyConnectedNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Sigmoid() #changable

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class ReducedNN(nn.Module):
    def __init__(self, original_nn, R):
        super(ReducedNN, self).__init__()
        self.input_dim   = original_nn.input_dim
        self.hidden_half = int(original_nn.hidden_dim/2)
        self.output_dim  = original_nn.output_dim
        self.fc1         = nn.Linear(self.input_dim, self.hidden_half)
        self.fc2         = nn.Linear(self.hidden_half, self.hidden_half)
        self.fc3         = nn.Linear(self.hidden_half, self.output_dim)

        # Transform weights and biases
        self.fc1.weight.data =  R @ original_nn.fc1.weight.data
        self.fc1.bias.data   = (R @ original_nn.fc1.bias.data.unsqueeze(1)).squeeze()

        self.fc2.weight.data =  R @ original_nn.fc2.weight.data @ R.T
        self.fc2.bias.data   = (R @ original_nn.fc2.bias.data.unsqueeze(1)).squeeze()

        self.fc3.weight.data = original_nn.fc3.weight.data @ R.T
        self.fc3.bias.data   = original_nn.fc3.bias.data

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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
        # loss_boundary = torch.mean(pde[self.boundary_mask])**2
        #regularization
        # loss_reg = self.reg_param * sum(param.abs().sum() for param in x.values())

        # loss = loss_pde+loss_boundary+loss_reg
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
    def __init__(self, NN_dim, n, mu, alpha, beta, n_samples = 1):
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
        self.mu    = mu



        self.NN    = FullyConnectedNN((n+1)**2, NN_dim, (n+1)**2)
        Res = restriction_R(int(self.NN.hidden_dim/2),self.NN.hidden_dim)
        self.NN.to(torch.float64)
        self.NN_low = ReducedNN(self.NN,Res)
        self.NN_low.to(torch.float64)
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
            #  'R':self.R,
            #  'Rlump':self.Rlump,
            #  'B':self.B,
             'k':self.kappa_value_list,
             'b':self.f_value_list,
             'ud':self.u_solution_tensor,
             'useEuclidean':useEuclidean,
             'mesh':torch.tensor(self.domain.geometry.x[:,0:1], dtype=torch.float64).T

            }
      return var
    def returnVars_low(self, useEuclidean):
      var = {'beta':self.beta,
             'n':self.n,
             'nu':self.n,
             'nz':sum(p.numel() for p in self.NN_low.parameters()),
             'alpha':self.alpha,
             'A':self.A,
             'M':self.M,
             'NN':self.NN_low,
            #  'R':self.R,
            #  'Rlump':self.Rlump,
            #  'B':self.B,
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
        X = geometry[:, 0]
        Y = geometry[:, 1]


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

#Subsolver for Taylor model
def trustregion_step_SPG2(x, val, dgrad, phi, problem, params, cnt):
    params.setdefault('maxitsp', 10)
    params.setdefault('lam_min', 1e-12)
    params.setdefault('lam_max', 1e12)
    params.setdefault('t', 1)
    params.setdefault('gtol', np.sqrt(np.finfo(float).eps))

    safeguard = 1e2 * np.finfo(float).eps

    x0 = x
    g0 = dgrad
    snorm = 0

    # Evaluate model at GCP
    sHs = 0
    gs = 0
    valold = val
    phiold = phi
    valnew = valold
    phinew = phiold

    t0 = max(params['lam_min'], min(params['lam_max'], params['t'] / problem.pvector.norm(dgrad)))

    # Set exit flag
    iter_count = 0
    iflag = 1

    for iter0 in range(1, params['maxitsp'] + 1):
        snorm0 = snorm

        # Compute step
        x1 = problem.obj_nonsmooth.prox(x0 - t0 * g0, t0)
        s = x1 - x0

        # Check optimality conditions
        gnorm = problem.pvector.norm(s)
        if gnorm / t0 <= params.get('tolsp', 0) and iter_count > 1:
            iflag = 0
            break

        # Compute relaxation parameter
        alphamax = 1
        snorm = problem.pvector.norm(x1 - x)
        if snorm >= params['delta'] - safeguard:
            ds = problem.pvector.dot(s, x0 - x)
            dd = gnorm ** 2
            alphamax = min(1, (-ds + np.sqrt(ds ** 2 + dd * (params['delta'] ** 2 - snorm0 ** 2))) / dd)

        #Hs = red_obj.hessVec(v, z, htol)[0]
        Hs = problem.obj_smooth.hessVec(s, x, params['gtol'])[0]

        sHs = problem.dvector.apply(Hs, s)
        g0s = problem.pvector.dot(g0, s)
        phinew = problem.obj_nonsmooth.value(x1)
        #eps = 1e-12
        alpha0 = max(-(g0s + phinew - phiold), gnorm ** 2 / t0) / sHs

        if sHs <= safeguard:
            alpha = alphamax
            if 0.5 * alphamax < alpha0 and iter0 > 1:
                alpha = 0
                phinew = phiold
                valnew = valold
                snorm = snorm0
                iflag = 3
                break
        else:
            alpha = min(alphamax, alpha0)

        # Update iterate
        if alpha == 1:
            x0 = x1
            g0 = problem.dvector.dual(Hs) + g0
            valnew = valold + g0s + 0.5 * sHs
        else:
            x0 = x0 + alpha * s
            g0 = alpha * problem.dvector.dual(Hs) + g0
            valnew = valold + alpha * g0s + 0.5 * alpha ** 2 * sHs
            phinew = problem.obj_nonsmooth.value(x0)
            snorm = problem.pvector.norm(x0 - x)

        # Update model information
        valold = valnew
        phiold = phinew

        # Check step size
        if snorm >= params['delta'] - safeguard:
            iflag = 2
            break

        norm_g0 = problem.pvector.norm(g0)

        # Update spectral step length
        if sHs <= safeguard:
            #if norm_g0 == 0:
                #norm_g0 = eps

            lambdaTmp = params['t'] / norm_g0

        else:
            lambdaTmp = gnorm ** 2 / sHs

        t0 = max(params['lam_min'], min(params['lam_max'], lambdaTmp))

    s = x0 - x
    pRed = (val + phi) - (valnew + phinew)

    iter_count = max(iter_count, iter0)
    return s, snorm, pRed, phinew, iflag, iter_count, cnt, params

def trustregion_step(l,x,val,grad,phi,problems,params,cnt):
    #fine level l comes in
    dgrad                   = problems[l].dvector.dual(grad) #dual(grad) puts in primal space
    L                       = len(problems)
    if l < L-1:
      R                       = problems[l].R
      Rdgrad                  = R @ dgrad
      Rgnorm                  = problems[l+1].pvector.norm(Rdgrad)
    else:
      Rgnorm = 0.0
    gnorm                   = problems[l].pvector.norm(dgrad)

    if Rgnorm > 0.5*gnorm and Rgnorm >= 1e-3: #note: counters are off
      problemsL = [] #problem list goes from fine to coarse
      for i in range(0, L):
        if i == l+1:
          p               = Problem(problems[i].obj_nonsmooth.var, problems[i].R) #make next level problem quadratic
          p.obj_smooth    = modelTR(problems, params["useSecant"], 'recursive', l = i, R = problems[i-1].R, grad = grad, x = x)
          p.obj_nonsmooth = phiPrec(problems[l], R = problems[i-1].R)
          # d = np.random.randn(R.shape[0],)
          # deriv_check(R @ x, d, p, 1e-4 * np.sqrt(np.finfo(float).eps))
          # stp
          problemsL.append(p)
        else:
          problemsL.append(problems[i])
      Deltai           = params['delta']
      xnew, cnt_coarse = trustregion(l+1, R @ x, Deltai, problemsL, params)
      #recompute snorm, pRed, phinew
      params['delta'] = Deltai
      s               = R.T @ xnew - x
      snorm           = problems[l].pvector.norm(s)
      #check L = f_{i-1} + <R g, s> + phi_{i-1}(x0 + s)
      #m = L + phi_i - phi_{i-1} so M = f_{i-1} + <Rg, s> + \phi_i
      phinew = problems[l].obj_nonsmooth.value(x + s)
      cnt['nobj2'] += 1
      valnew, _ = problemsL[l+1].obj_smooth.value(xnew, 0.0)
      pRed   = val + phi - valnew - phinew
      iflag  = cnt_coarse['iflag']
      iter   = cnt_coarse['iter']
      cnt    = problemsL[l+1].obj_smooth.addCounter(cnt)
      cnt    = problemsL[l+1].obj_nonsmooth.addCounter(cnt)
    else:
      R                       = np.eye(x.shape[0])
      problemTR               = Problem(problems[l].var, R)
      problemTR.obj_smooth    = modelTR(problems, params["useSecant"], 'spg', l = l, R = R, grad = grad, x = x)
      problemTR.obj_nonsmooth = phiPrec(problems[l], R = R)
      # d = np.random.randn((x.shape[0]))
      # deriv_check(x, d, problemTR, 1e-4 * np.sqrt(np.finfo(float).eps))
      s, snorm, pRed, phinew, iflag, iter, cnt, params = trustregion_step_SPG2(x, val, dgrad, phi, problemTR, params, cnt)
      cnt = problemTR.obj_smooth.addCounter(cnt)
      cnt = problemTR.obj_nonsmooth.addCounter(cnt)
    return s, snorm, pRed, phinew, iflag, iter, cnt, params

class modelTR:
    def __init__(self, problems, secant, subtype = 'spg', l = 0, R = np.empty(1), grad = np.empty(1), x = np.empty(1)):
        self.problem = problems[l]
        self.var     = problems[l].var
        self.secant  = secant
        self.l       = l
        self.x       = R @ x
        self.R       = R
        self.Rgrad   = problems[l].pvector.dual(R @ grad) #should be in dual space, dgrad in primal
        self.subtype = subtype
        self.nobj1   = 0
        self.ngrad   = 0
        self.nhess   = 0
        if subtype == 'recursive':
            grad, _      = problems[l].obj_smooth.gradient(R @ x, 0.)
            self.grad    = grad
            self.ngrad  += 1

    def update(self, x, type):
        self.problem.obj_smooth.update(x, type)
    def value(self, x, ftol):
        val, ferr    = self.problem.obj_smooth.value(x, ftol)
        if self.subtype == 'recursive':
          # val      += self.problem.dvector.apply(self.Rgrad - 0.*self.grad, x - self.x)
          val      += self.problem.dvector.apply(self.Rgrad, x - self.x)
          ferr      = 0
        self.nobj1 += 0
        return val, ferr
    def gradient(self,x,gtol):
      grad, gerr      = self.problem.obj_smooth.gradient(x, gtol)
      if self.subtype == 'recursive':
        # grad        += self.Rgrad - 0.*self.grad
        grad        += self.Rgrad
        self.ngrad += 0
      return grad, gerr
    def hessVec(self,v,x,htol):
      if (self.secant):
        hv          = self.problem.secant.apply(v,self.problem.pvector,self.problem.dvector)
        herr        = 0
      else:
        hv, herr    = self.problem.obj_smooth.hessVec(v, x, htol)
        self.nhess += 1
      return hv, herr
    def addCounter(self,cnt):
        #actually zero because you should eval in subprob?
        cnt["nobj1"] += self.nobj1
        cnt["ngrad"] += self.ngrad
        cnt["nhess"] += self.nhess
        return cnt

class phiPrec: # you can definitely clean this up and inherit a bunch of stuff but
               # we can be explicit for now
    def __init__(self, problem, R = np.empty(1)):
        self.problem   = problem
        self.var       = problem.obj_nonsmooth.var
        self.R         = R
        self.nobj2     = 0
        self.nprox     = 0
    def value(self, x):
        val             = self.problem.obj_nonsmooth.value(self.R.T @ x)
        self.nobj2     += 1
        return val
    def prox(self, x, t):
        px          = x + self.R @ (self.problem.obj_nonsmooth.prox(self.R.T @ x, t) - self.R.T @ x)
        self.nprox += 1
        return px
    def addCounter(self, cnt):
        cnt["nobj2"] += self.nobj2
        cnt["nprox"] += self.nprox
        return cnt
    def genJacProx(self, x, t):
        D, ind = self.problem.obj_nonsmooth.genJacProx(x, t)
        return D, ind
    def applyProxJacobian(self, v, x, t):
        Dv = self.problem.obj_nonsmooth.applyProxJacobian(v, x, t)
        return Dv
    def getParameter(self):
        return self.problem.obj_nonsmooth.getParameter()

def trustregion(l, x0, Deltai, problems, params): #inpute Deltai
    """
    Trust-region optimization algorithm.

    Parameters:
    x0 (np.array): Initial guess for the control variable.
    problem (Problem): Problem class containing objective functions and vector spaces.
    params (dict): Dictionary of algorithmic parameters.

    Returns:
    x (np.array): Optimized control variable.
    cnt (dict): Dictionary of counters and history.
    """
    start_time = time.time()

    # Check and set default parameters
    params.setdefault('outFreq', 1)
    params.setdefault('initProx', False)
    params.setdefault('t', 1.0)
    params.setdefault('maxit', 100)
    params.setdefault('reltol', False)
    params.setdefault('gtol', 1e-6)
    params.setdefault('stol', 1e-6)
    params.setdefault('ocScale', 1.0)
    params.setdefault('atol', 1e-4)
    params.setdefault('rtol', 1e-2)
    params.setdefault('spexp', 2)
    params.setdefault('eta1', 1e-4)
    params.setdefault('eta2', 0.75)
    params.setdefault('gamma1', 0.25)
    params.setdefault('gamma2', 10.0)
    params.setdefault('delta', 1.0)
    params.setdefault('deltamax', 1e10)
    params.setdefault('useSecant', False)
    params.setdefault('useSecantPrecond', False)
    params.setdefault('secantType', 2)
    params.setdefault('secantSize', 10)
    params.setdefault('initScale', 1.0)
    params.setdefault('useDefault', True)
    params.setdefault('useInexactObj', False)
    params.setdefault('etascale', 1 - 1e-3)
    params.setdefault('maxValTol', 1)
    params.setdefault('scaleValTol', 1e0)
    params.setdefault('forceValTol', 1 - 1e-3)
    params.setdefault('expValTol', 0.9)
    params.setdefault('useInexactGrad', False)
    params.setdefault('maxGradTol', 1)
    params.setdefault('scaleGradTol', 1e0)

    # Initialize counters
    cnt = {
        'AlgType': f"TR-{params.get('spsolver', 'SPG2')}",
        'iter': 0,
        'nobj1': 0,
        'ngrad': 0,
        'nhess': 0,
        'nobj2': 0,
        'nprox': 0,
        'nprec': 0,
        'timetotal': 0,
        'objhist': [],
        'obj1hist': [],
        'obj2hist': [],
        'gnormhist': [],
        'snormhist': [],
        'deltahist': [],
        'nobj1hist': [],
        'nobj2hist': [],
        'ngradhist': [],
        'nhesshist': [],
        'nproxhist': [],
        'timestor': [],
        'valerr': [],
        'valtol': [],
        'graderr': [],
        'gradtol': []
    }

    # Compute initial function information
    if hasattr(problems[l].obj_smooth, 'begin_counter'):
        cnt = problems[l].obj_smooth.begin_counter(0, cnt)

    if params['initProx']:
        x = problems[l].obj_nonsmooth.prox(x0, 1)
        cnt['nprox'] += 1
    else:
        x = x0

    problems[l].obj_smooth.update(x, 'init')
    ftol = 1e-12
    if params['useInexactObj']:
        ftol = params['maxValTol']
    params['delta'] = min(params['delta'], Deltai)
    val, _      = problems[l].obj_smooth.value(x, ftol)

    cnt['nobj1'] += 1

    grad, _, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
    phi = problems[l].obj_nonsmooth.value(x)
    cnt['nobj2'] += 1

    if hasattr(problems[l].obj_smooth, 'end_counter'):
        cnt = problems[l].obj_smooth.end_counter(0, cnt)

    if params['useSecantPrecond']:
        problems[l].prec.apply         = lambda x: problems[l].secant.apply(x, problems[l].pvector, problems[l].dvector)
        problems[l].prec.apply_inverse = lambda x: problems[l].secant.apply_inverse(x, problems[l].pvector, problems[l].dvector)

    # Output header
    if l == 0:
      print(f"\nRecursive Nonsmooth Trust-Region Method using {params.get('spsolver', 'SPG2')} Subproblem Solver")
      print("level   iter          value           gnorm             del           snorm       nobjs      ngrad      nhess      nobjn      nprox    iterSP    flagSP")
      print(f"{0:4d}   {0:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}             ---      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}       ---      ---")

    # Storage
    cnt['objhist'].append(val + phi)
    cnt['obj1hist'].append(val)
    cnt['obj2hist'].append(phi)
    cnt['gnormhist'].append(gnorm)
    cnt['snormhist'].append(np.nan)
    cnt['deltahist'].append(params['delta'])
    cnt['nobj1hist'].append(cnt['nobj1'])
    cnt['nobj2hist'].append(cnt['nobj2'])
    cnt['ngradhist'].append(cnt['ngrad'])
    cnt['nhesshist'].append(cnt['nhess'])
    cnt['nproxhist'].append(cnt['nprox'])
    cnt['timestor'].append(np.nan)

    # Set optimality tolerance
    gtol = params['gtol']
    stol = params['stol']
    if params['reltol']:
        gtol = params['gtol'] * gnorm
        stol = params['stol'] * gnorm

    # Check stopping criterion
    if gnorm <= gtol:
        cnt['iflag'] = 0
        return x, cnt

    # Iterate
    for i in range(1, params['maxit'] + 1):
        if hasattr(problems[l].obj_smooth, 'begin_counter'):
            cnt = problems[l].obj_smooth.begin_counter(i, cnt)

        # Solve trust-region subproblem
        params['tolsp'] = min(params['atol'], params['rtol'] * gnorm ** params['spexp'])
        s, snorm, pRed, phinew, iflag, iter_count, cnt, params = trustregion_step(l, x, val, grad, phi, problems, params, cnt)

        # Update function information
        xnew = x + s
        problems[l].obj_smooth.update(xnew, 'trial')
        valnew, val, cnt = compute_value(xnew, x, val, problems[l].obj_smooth, pRed, params, cnt)

        # Accept/reject step and update trust-region radius
        aRed = (val + phi) - (valnew + phinew)
        if aRed < params['eta1'] * pRed:
            params['delta'] = params['gamma1'] * min(snorm, params['delta'])
            problems[l].obj_smooth.update(x, 'reject')
            if params['useInexactGrad']:
                grad, dgrad, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
        else:
            x = xnew
            val = valnew
            phi = phinew
            problems[l].obj_smooth.update(x, 'accept')
            grad0 = grad
            grad, dgrad, gnorm, cnt = compute_gradient(x, problems[l], params, cnt)
            if aRed > params['eta2'] * pRed:
                params['delta'] = min(params['deltamax'], params['gamma2'] * params['delta'])
                params['delta'] = min(params['delta'], Deltai - problems[l].pvector.norm(x - x0))
            # Update secant
            if params['useSecant'] or params['useSecantPrecond']:
                y = grad - grad0
                problems.secant.update(s, y, problems[l].pvector, problems[l].dvector)
                if params['useSecantPrecond']:
                    problems[l].prec.apply = lambda x: problems[l].secant.apply(x, problems[l].pvector, problems[l].dvector)
                    problems[l].prec.apply_inverse = lambda x: problems[l].secant.apply_inverse(x, problems[l].pvector, problems[l].dvector)

        # Output iteration history
        if i % params['outFreq'] == 0:
            print(f"{l:4d}   {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")

        # Storage
        cnt['objhist'].append(val + phi)
        cnt['obj1hist'].append(val)
        cnt['obj2hist'].append(phi)
        cnt['gnormhist'].append(gnorm)
        cnt['snormhist'].append(snorm)
        cnt['deltahist'].append(params['delta'])
        cnt['nobj1hist'].append(cnt['nobj1'])
        cnt['nobj2hist'].append(cnt['nobj2'])
        cnt['ngradhist'].append(cnt['ngrad'])
        cnt['nhesshist'].append(cnt['nhess'])
        cnt['nproxhist'].append(cnt['nprox'])
        cnt['timestor'].append(time.time() - start_time)

        if hasattr(problems[l].obj_smooth, 'end_counter'):
            cnt = problems[l].obj_smooth.end_counter(i, cnt)

        # Check stopping criterion
        if gnorm <= gtol or snorm <= stol or i >= params['maxit']:
            if i % params['outFreq'] != 0:
                print(f"  {l:4d}   {i:4d}    {val + phi:8.6e}    {gnorm:8.6e}    {params['delta']:8.6e}    {snorm:8.6e}      {cnt['nobj1']:6d}     {cnt['ngrad']:6d}     {cnt['nhess']:6d}     {cnt['nobj2']:6d}     {cnt['nprox']:6d}      {iter_count:4d}        {iflag:1d}")
            if gnorm <= gtol:
                flag = 0
            elif i >= params['maxit']:
                flag = 1
            else:
                flag = 2
            break

    cnt['iter'] = i
    cnt['timetotal'] = time.time() - start_time

    print("Optimization terminated because ", end="")
    if flag == 0:
        print("optimality tolerance was met")
    elif flag == 1:
        print("maximum number of iterations was met")
    else:
        print("step tolerance was met")
    print(f"Total time: {cnt['timetotal']:8.6e} seconds")
    cnt['iflag'] = flag
    return x, cnt

def compute_value(x, xprev, fvalprev, obj, pRed, params, cnt):
    """
    Compute the objective function value with inexactness handling.

    Parameters:
    x (np.array): Current point.
    xprev (np.array): Previous point.
    fvalprev (float): Previous function value.
    obj (Objective): Objective function class.
    pRed (float): Predicted reduction.
    params (dict): Algorithm parameters.
    cnt (dict): Counters.

    Returns:
    fval (float): Current function value.
    fvalprev (float): Previous function value.
    cnt (dict): Updated counters.
    """
    ftol = 1e-12
    valerrprev = 0
    if params['useInexactObj']:
        omega = params['expValTol']
        scale = params['scaleValTol']
        force = params['forceValTol']
        eta = params['etascale'] * min(params['eta1'], 1 - params['eta2'])
        ftol = min(params['maxValTol'], scale * (eta * min(pRed, force ** cnt['nobj1'])) ** (1 / omega))
        fvalprev, valerrprev = obj.value(xprev, ftol)
        cnt['nobj1'] += 1

    obj.update(x, 'trial')
    fval, valerr = obj.value(x, ftol)
    cnt['nobj1'] += 1
    cnt['valerr'].append(max(valerr, valerrprev))
    cnt['valtol'].append(ftol)

    return fval, fvalprev, cnt

def compute_gradient(x, problem, params, cnt):
    """
    Compute the gradient with inexactness handling.

    Parameters:
    x (np.array): Current point.
    problem (Problem): Problem class.
    params (dict): Algorithm parameters.
    cnt (dict): Counters.

    Returns:
    grad (np.array): Gradient.
    dgrad (np.array): Dual gradient.
    gnorm (float): Gradient norm.
    cnt (dict): Updated counters.
    """
    if params['useInexactGrad']:
        scale0 = params['scaleGradTol']
        gtol = min(params['maxGradTol'], scale0 * params['delta'])
        gerr = gtol + 1
        while gerr > gtol:
            grad, gerr = problem.obj_smooth.gradient(x, gtol)
            cnt['ngrad'] += 1
            dgrad = problem.dvector.dual(grad)
            pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
            cnt['nprox'] += 1
            gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']
            gtol = min(params['maxGradTol'], scale0 * min(gnorm, params['delta']))
    else:
        gtol = 1e-12
        grad, gerr = problem.obj_smooth.gradient(x, gtol)
        cnt['ngrad'] += 1
        dgrad = problem.dvector.dual(grad)
        pgrad = problem.obj_nonsmooth.prox(x - params['ocScale'] * dgrad, params['ocScale'])
        cnt['nprox'] += 1
        gnorm = problem.pvector.norm(pgrad - x) / params['ocScale']

    params['gradTol'] = gtol
    cnt['graderr'].append(gerr)
    cnt['gradtol'].append(gtol)

    return grad, dgrad, gnorm, cnt

def driver(savestats, name):
    print("driver started")
    np.random.seed(0)

    # Set up optimization problem
    n          = 3  # Number of cells
    NN_dim     = 100 # Neural network nodes
    nu         = 0.08  # Viscosity
    alpha      = 1  # L2 penalty parameter
    beta       = 1e-4  # L1 penalty parameter
    derivCheck = True
    R_res = restriction_R(50,100)

    nnset = NNSetup(NN_dim, n, nu, alpha, beta, n_samples=1)
    problems = [] #problem list goes from fine to coarse
    var   = nnset.returnVars(False)
    problem = Problem(var)

    if derivCheck:
        x =  TorchVect(nnset.NN.state_dict())
        d = x.clone() #just to get something to multiply against
        for key, vals in d.td.items():
          d.td[key] = vals.copy_(torch.randn(vals.size()))
        deriv_check(x, d, problem, 1e-4 * np.sqrt(np.finfo(float).eps))
        vector_check(x, d, problem)

    x0 =  TorchVect(nnset.NN.state_dict())
    x0low = TorchVect(nnset.NN_low.state_dict())
    cnt = {}

    # Update default parameters
    params = set_default_parameters("SPG2")
    params["reltol"] = False
    params["t"] = 2 / alpha
    params["ocScale"] = 1 / alpha

    # Solve optimization problem
    start_time   = time.time()
    x, cnt_tr    = trustregion(R_res, x0low, x0, problem_low,problem, params)
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
    nnset.NN.load_state_dict(x.td)  # Load the updated parameters into the neural network
    nnset.NN.eval()
    state_dict_after  = nnset.NN.state_dict()
    weights_fc1_after = state_dict_after['fc1.weight']
    print("Weights of fc1 after optimization:", torch.nonzero(weights_fc1_after))

    # Use the updated neural network for inference or further tasks
    updated_nn = nnset.NN
    print("Updated neural network is stored in `updated_nn`.")
    nnset.plot_solutions()

    return cnt


cnt = driver(False, "test_run")
