import numpy as np
import scipy.sparse as sp

from scipy.sparse import spdiags, diags, lil_matrix,block_diag, kron, eye
from scipy.integrate import quad
import ufl, dolfinx
from dolfinx import fem, mesh,  plot,la
from mpi4py import MPI
#import dolfinx.fem.petsc
#import dolfinx.nls.petsc
import os
import sys
sys.path.append(os.path.abspath("/Users/wang/Multilevel_LM/Multilevel_LM"))

from dolfinx.fem.petsc import NonlinearProblem, LinearProblem
from dolfinx.nls.petsc import NewtonSolver
from petsc4py import PETSc
from non_smooth.checks import deriv_check, deriv_check_simopt, vector_check
from non_smooth.setDefaultParameters import set_default_parameters
import time
from dolfinx.mesh import create_unit_square,create_rectangle,CellType
#from non_smooth.L1norm import L1Norm
from non_smooth.trustregion import trustregion
import matplotlib.pyplot as plt

class SemilinearEllipticSetup:
    """
    solve the semilinear elliptic pdes
    
    """
    def __init__(self,n,alpha,beta):
        if n <= 1:
            raise ValueError("Number of cells (n) must be greater than 1.")
            
        if alpha <= 0:
            raise ValueError("Control penalty parameter (alpha) must be positive.")
            
        
        #self.domain = create_unit_square(MPI.COMM_WORLD, n, n)
        self.domain = create_rectangle(
             MPI.COMM_WORLD,
             points=((0.0, 0.0), (1.0, 1.0)),
             n=(n, n),
             cell_type=CellType.quadrilateral,
             ghost_mode = mesh.GhostMode.shared_facet
        )
        
        #self.V = fem.functionspace(self.domain,("DG",0))
        self.V = fem.functionspace(self.domain,("Lagrange",1))
        self.alpha = alpha
        self.beta = beta
        self.n = n
        
        #boundary condition (homogeneous Dirichlet)
        tdim = self.domain.topology.dim
        self.domain.topology.create_connectivity(tdim-1,tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        boundary_dofs = fem.locate_dofs_topological(self.V, tdim-1, boundary_facets)
        self.bc = fem.dirichletbc(PETSc.ScalarType(0),boundary_dofs,self.V)
        
        #Target state (desired solution)
        self.y_d = fem.Function(self.V)
        x = ufl.SpatialCoordinate(self.domain)
        y_d_expr = ufl.conditional(
            ufl.lt(x[0],0.5),
            ((x[0]-0.5)**4+0.5*(x[1]-0.5)**3)*ufl.sin(ufl.pi*x[1]),
            0.0
        )
        #y_d_expr = ufl.sin(np.pi * x[1]) * ufl.cos(np.pi * x[0])
        #y_d = fem.Function(self.V)
        #self.y_d.x.array[:] = 1.0  # Set all DOFs directly to 1
        #self.y_d.x.scatter_forward()

        #self.y_d.interpolate(y_d_expr)
        
        #y_d.interpolate(lambda x: np.full(x.shape[1], 1.0))
        #self.y_d.interpolate(lambda x: np.ones(x.shape[1]))
        #y_d_expr = x[0]*(x[1]-1)*x[1]*(x[0]-1)
        self.y = fem.Function(self.V)
        self.u = fem.Function(self.V)
        
        self.p = fem.Function(self.V)
        #self.y_d = y_d
        #interpolation_points = self.V.element.interpolation_points()
        #y_d_expr_proj = fem.Expression(y_d_expr,interpolation_points)
        self.y_d.interpolate(fem.Expression(y_d_expr,self.V.element.interpolation_points()))
        #self.y_d.x.scatter_forward()
        # After setup, verify the target function
        
        #self.y_d.interpolate(fem.Expression(y_d_expr,self.V.element.interpolation_points()))
        self.reduced_obj = ReducedObjective(self)
        #print("Mesh cells (expected 32×32 = 1024):", self.domain.topology.index_map(self.domain.topology.dim).size_local)
        #print("Function space V DoFs:", self.V.dofmap.index_map.size_local)
        
    def solve_state_equation(self):
        """
        Solve -Delta y + y^3 - u = 0 with y = 0 on boundary

        """
        v = ufl.TestFunction(self.V)
        F = (ufl.inner(ufl.grad(self.y),ufl.grad(v))+self.y**3*v-self.u*v)*ufl.dx
        x = ufl.SpatialCoordinate(self.domain)
        self.y.interpolate(fem.Expression(0.1*ufl.sin(np.pi*x[0])*ufl.sin(np.pi*x[1]),self.V.element.interpolation_points()))
        PETSc.Options()["snes_rtol"] = 1e-12
        PETSc.Options()["snes_atol"] = 1e-12
        PETSc.Options()["snes_max_it"] = 50
        PETSc.Options()["snes_monitor"] = None
        PETSc.Options()["snes_linesearch_type"] = "bt"

        problem = NonlinearProblem(F,self.y,bcs = [self.bc])
        solver = NewtonSolver(MPI.COMM_WORLD,problem)
        
        
        # Initialize y
        self.y.x.array[:] = 0.0

        # Function to store the residual
        residual = fem.Function(self.V)
        

        # Solve and monitor residual
        num_iters, converged = solver.solve(self.y)
        

        # Compute the final residual norm (if needed)
        residual_vector = la.create_petsc_vector(self.V.dofmap.index_map, self.V.dofmap.index_map_bs)
    
        # Recreate the residual form using the test function
        residual_form = fem.form(ufl.replace(F, {self.y: self.y, v: ufl.TestFunction(self.V)}))
    
        # Assemble the residual
        with residual_vector.localForm() as local_residual:
            local_residual.set(0.0)
            fem.petsc.assemble_vector(local_residual, residual_form)
        residual_vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    
        # Compute norm
        residual_norm = residual_vector.norm()
        print(f"Final residual norm: {residual_norm:.3e}")

        
        
    def solve_adjoint_equation(self):
        """
        
        Solve the adjoint equation -Delta p +3y^2 p = -(y-y_d) with p=0 on boundary
        
        """
        V = self.V
        p = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(ufl.grad(p),ufl.grad(v))*ufl.dx+3*self.y**2*p*v*ufl.dx
        #L = -ufl.inner(self.y-self.y_d,v)*ufl.dx
        L = (self.y - self.y_d) * v * ufl.dx
        problem = LinearProblem(a,L,bcs=[self.bc],petsc_options={"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10})
        self.p = problem.solve()
        #problem.solve(self.p)
        
    
    # Vectorized control update
    def update_control(self):
        p = self.p.x.array
        u_new = np.zeros_like(p)
        mask_pos = p > self.beta
        mask_neg = p < -self.beta
        u_new[mask_pos] = (p[mask_pos] - self.beta)/self.alpha
        u_new[mask_neg] = (p[mask_neg] + self.beta)/self.alpha
        self.u.x.array[:] = u_new
        
    def returnVars(self,useEuclidean=False):
        
        var={
            'V':self.V,
            #'Vc':self.Vc,
            'alpha':self.alpha,
            'beta': self.beta,
            'y_d':self.y_d,
            'bc':self.bc,
            'domain': self.domain,
            'useEuclidean': useEuclidean,
            'setup':self,
            'n':self.n
            }
        return var
        
            
            

class ReducedObjective:
    """
    Reduced objective for the PDE-constrained optimal problem:
        J(u,z) = 0.5*||y-yd||**2+0.5*alpha*||u||**2+beta*||u||_1
        s.t. -\Delta y + y**3 - u = 0 and y = 0 on the boundary 
    
    """
    
    
    
    def __init__(self,setup:SemilinearEllipticSetup):
        
        self.setup = setup
        self.is_state_computed = False
        self.is_adjoint_computed = False
        self.cnt = {
            'nstate': 0,
            'nadjoint': 0,
            'nstatesens': 0,
            'nadjointsens': 0,
            'ninvJ1': 0
            }
        self.V = setup.V
        self.u = setup.u
        self.y = setup.y
        self.p = setup.p
        
    def begin_counter(self, iter, cnt0):
        if iter == 0:
            cnt0.update({
                'nstatehist': [],
                'nadjoihist': [],
                'nstsenhist': [],
                'nadsenhist': [],
                'ninvJ1hist': []
            })
            for key in self.cnt:
                self.cnt[key] = 0
        return cnt0

    def end_counter(self, iter, cnt0):
        cnt0['nstatehist'].append(self.cnt['nstate'])
        cnt0['nadjoihist'].append(self.cnt['nadjoint'])
        cnt0['nstsenhist'].append(self.cnt['nstatesens'])
        cnt0['nadsenhist'].append(self.cnt['nadjointsens'])
        cnt0['ninvJ1hist'].append(self.cnt['ninvJ1'])
        return cnt0

    def reset(self):
        self.is_state_computed = False
        self.is_adjoint_computed = False
        
    def update(self, u, update_type):
        if update_type == 'init':
            self.is_state_computed = False
            self.is_adjoint_computed = False
        elif update_type == 'trial':
            self.is_state_computed = False
            self.is_adjoint_computed = False
        elif update_type == 'accept':
            pass 
        
        
    def value(self,u,ftol=1e-12):
        """
        Evaluate the reduced cost function J(u)

        """
        if not self.is_state_computed:
            self.setup.u.x.array[:] = u
            self.setup.solve_state_equation()
            self.cnt['nstate'] += 1
            self.is_state_computed = True
            
        #self.setup.solve_state_equation()
        y = self.y
        u_func = fem.Function(self.setup.V)
        u_func.x.array[:] = u
        
        
        y_d = self.setup.y_d 
        alpha = self.setup.alpha
        beta = self.setup.beta
        
        
        J1 = 0.5*fem.assemble_scalar(fem.form((y-y_d)**2*ufl.dx))
        J2 = 0.5*alpha*fem.assemble_scalar(fem.form(u_func**2*ufl.dx(self.setup.domain)))
        
        
        J3 = beta*fem.assemble_scalar(fem.form(ufl.algebra.Abs(self.u)*ufl.dx))
        return J1+J2+J3,0
    
    def gradient(self,u,gtol=1e-12):
        """
        Compute reduced gradient \nabla J(u)=alpha*u+ beta*sign(u)+p
        
        """
        
        alpha = self.setup.alpha
        beta = self.setup.beta
        
        
        if not self.is_state_computed:
            self.setup.u.x.array[:] = u
            self.setup.solve_state_equation()
            self.cnt['nstate'] += 1
            self.is_state_computed = True

        if not self.is_adjoint_computed:
            self.setup.solve_adjoint_equation()
            self.cnt['nadjoint'] += 1
            self.is_adjoint_computed = True
            
        
        

        grad = (alpha * u + 
                beta * np.sign(u) +  # L1 term
                self.setup.p.x.array)
        
        
        return grad,0.0
    
    def hessVec(self,v,u,htol=1e-12):
        """
        
        Compute reduced Hessian-vector product J"(u)*v

        """
        if not self.is_state_computed:
            self.setup.u.x.array[:] = u
            self.setup.solve_state_equation()
            self.cnt['nstate'] += 1
            self.is_state_computed = True

        if not self.is_adjoint_computed:
            self.setup.solve_adjoint_equation()
            self.cnt['nadjoint'] += 1
            self.is_adjoint_computed = True
        V = self.setup.V
        alpha= self.setup.alpha
        v_func = fem.Function(V)
        
        v_func.x.array[:] = v
        
        
        #Solve linearized state equation: -\Delta (dy) + 3y^2 dy = du
        dy = fem.Function(V)
        dy_trial = ufl.TrialFunction(V)
        w_test = ufl.TestFunction(V)
        
        a_state = (ufl.inner(ufl.grad(dy_trial),ufl.grad(w_test))+3*self.y**2*dy_trial*w_test)*ufl.dx(self.setup.domain)
        L_state = v_func*w_test*ufl.dx(self.setup.domain)
        problem_dy = LinearProblem(a_state,L_state,bcs=[self.setup.bc])
        dy = problem_dy.solve()
        self.cnt['nstatesens'] += 1
        
        #Solve linearized adjoint equation: -\Delta(dp) +3y^2 dp = -2*y*dy
        dp = fem.Function(V)
        dp_trial = ufl.TrialFunction(V)
        
        a_adj = (ufl.inner(ufl.grad(dp_trial),ufl.grad(w_test))+3*self.y**2*dp_trial*w_test)*ufl.dx(self.setup.domain)
        L_adj = -2 * (self.setup.y - self.setup.y_d) * dy * w_test * ufl.dx(self.setup.domain)
        problem_dp = LinearProblem(a_adj,L_adj,bcs=[self.setup.bc])
        dp = problem_dp.solve()
        self.cnt['nadjointsens'] += 1
        
        #Reduced Hessian action alpha*du+dp
        Hv = (alpha*v+dp.x.array)
        
        return Hv,0.0
    def profile(self):
        print("\nProfile Reduced Objective")
        print("  #state    #adjoint    #statesens    #adjointsens    #linearsolves")
        print(f"  {self.cnt['nstate']:6d}      {self.cnt['nadjoint']:6d}        {self.cnt['nstatesens']:6d}          {self.cnt['nadjointsens']:6d}           {self.cnt['ninvJ1']:6d}")
        return self.cnt.copy()
    
    
    
    
    
    
def restriction_R_Reduced_2d(m,n):
    """
    Create 2D restriction matrix for u in 2D
     
    m: coarse dimension
    n: fine dimension

    """
    r_1d = lil_matrix((m,n))
    ratio = n//m
    for i in range(m):
        r_1d[i,i*ratio:(i+1)*ratio] = 1/(np.sqrt(ratio))
        
    r_2d = kron(r_1d,r_1d).tocsc()
    return r_2d   
    
    
    
    
class PDEProblemWrapper:
    def __init__(self, setup, R):
        self.setup = setup
        self.control_dim= (setup.n+1)**2
        self.obj_smooth = PDEObjective(setup)
        self.obj_nonsmooth = L1NormRegularizer(setup)
        self.pvector = L2ControlPrimal(setup)
        self.dvector = L2ControlDual(setup)
        self.R = R

class PDEObjective:
    def __init__(self, setup):
        self.setup = setup
        
    def update(self, u, type):
        return None
        
    def value(self, u, fol=1e-12):
        self.setup.u.x.array[:] = u
        #return self.setup.reduced_obj.value(), 0  # (value, error_estimate)
        self.setup.solve_state_equation()
        y = self.setup.y
        y_d = self.setup.y_d
        alpha = self.setup.alpha
        u_func = fem.Function(self.setup.V)
        u_func.x.array[:] = u
        J1 = 0.5*fem.assemble_scalar(fem.form((y-y_d)**2*ufl.dx(self.setup.domain)))
        J2 = 0.5*alpha*fem.assemble_scalar(fem.form(u_func**2*ufl.dx(self.setup.domain)))
        return J1+J2 , 0
    
    def gradient(self, u, Gol=1e-12):
        self.setup.u.x.array[:] = u
        self.setup.solve_state_equation()
        self.setup.solve_adjoint_equation()
        grad = (self.setup.alpha * u + self.setup.p.x.array)
        return grad, 0  # (grad, error_estimate)
    
    def hessVec_11(self,v, u, htol=1e-12):
        self.setup.u.x.array[:] = u
        self.setup.solve_state_equation()
        # Linearized state solve
        dy = fem.Function(self.setup.V)
        dy_trial = ufl.TrialFunction(self.setup.V)
        w_test = ufl.TestFunction(self.setup.V)
        v_func = fem.Function(self.setup.V)
        v_func.x.array[:] = v
        
        a_state = (ufl.inner(ufl.grad(dy_trial), ufl.grad(w_test)) + 3*self.setup.y**2*dy_trial*w_test)*ufl.dx(self.setup.domain)
        L_state = v_func * w_test * ufl.dx(self.setup.domain)
        problem_dy = LinearProblem(a_state,L_state,bcs=[self.setup.bc])
        dy = problem_dy.solve()
        
        # Linearized adjoint solve
        dp = fem.Function(self.setup.V)
        dp_trial = ufl.TrialFunction(self.setup.V)
        a_adj = (ufl.inner(ufl.grad(dp_trial), ufl.grad(w_test)) + 3*self.setup.y**2*dp_trial*w_test)*ufl.dx(self.setup.domain)
        L_adj = -2 * (self.setup.y - self.setup.y_d) * dy_trial * w_test * ufl.dx(self.seup.domain)
        problem_dp = LinearProblem(a_adj,L_adj,bcs=[self.setup.bc])
        dp = problem_dp.solve()
        
        Hv = self.setup.alpha * v + dp.x.array
        return Hv, 0 
    def hessVec_12(self, v, u, htol=1e-12):
        """Zero for control-state coupling (if needed)"""
        return np.zeros_like(u), 0

    def hessVec_21(self, v, u, htol=1e-12):
        """Zero for state-control coupling (if needed)"""
        return np.zeros_like(u), 0

    def hessVec_22(self, v, u, htol=1e-12):
        """Zero for state-state part (if needed)"""
        return np.zeros_like(u), 0
        
        

class L1NormRegularizer:
    def __init__(self, setup_or_var):
        if isinstance(setup_or_var,dict):
            self.var = setup_or_var
            self.h = 1/setup_or_var['setup'].n
        else:
            self.var = {'setup':setup_or_var}
            self.h = 1/setup_or_var.n
        self.beta = setup_or_var.beta
        
    def value(self, u):
        return self.beta * np.sum(np.abs(u))
        
    def prox(self, u, t):
        """Proximal operator for L1 norm"""
        return np.sign(u) * np.maximum(np.abs(u) - t*self.beta, 0)
    def get_parameter(self):
        return self.beta
    
class L2ControlPrimal:
    """Implements the L2 primal vector space operations for control variables"""
    
    def __init__(self, setup_or_var):
        if isinstance(setup_or_var,dict):
            self.var = setup_or_var
            self.h = 1/setup_or_var['setup'].n
        else:
            self.var = {'setup':setup_or_var}
            self.h = 1/setup_or_var.n
        
    
    def apply(self, u, v):
        return self.dot(u,v)
    
    def dual(self,u):
        return u.copy()
        
        
    def norm(self, u):
        """Compute L2 norm of control vector"""
        return np.sqrt(self.dot(u, u))
        
    def dot(self, u, v):
        """L2 inner product with mesh scaling"""
        return self.h**2 * np.dot(u, v)  # h^2 for 2D quadrature
        
    def proj(self, u):
        """Identity projection for unconstrained controls"""
        return u.copy()
    
class L2ControlDual:
    """Implements dual space operations for control variables"""
    
    def __init__(self, setup_or_var):
        if isinstance(setup_or_var,dict):
            self.var = setup_or_var
            self.h = 1/setup_or_var['setup'].n
        else:
            self.var = {'setup':setup_or_var}
            self.h = 1/setup_or_var.n
        
        
    def dual(self, u):
        """Convert primal to dual representation (identity for L2)"""
        return u.copy()
    
    def apply(self, u, v):
        return self.dot(u,v)
        

    def primal(self, u):
        """Convert dual to primal representation"""
        return u.copy()
        
    def norm(self, u):
        """Dual norm computation"""
        return np.sqrt(self.dot(u, u))
        
    def dot(self, u, v):
        """Dual inner product"""
        return (1.0/self.h**2) * np.dot(u, v)  # Inverse L2 metric    
    
from types import SimpleNamespace    
def driver_pde(savestats=False, name="pde_optim"):
    print("\n=== PDE-Constrained Optimization Driver ===")
    np.random.seed(0)
    

    # Problem parameters
    #n_levels = [32, 16]  # Fine to coarse mesh sizes
    n_levels = [64, 32]
    alpha = 1e-4         # L2 regularization
    beta = 1e-6          # L1 regularization
    
    params = set_default_parameters("SPG2")
    params['maxit']   = 500
    params['reltol'] = False
    params['t'] = 2/alpha
    params['ocScale'] = 1/alpha
    params['gtol']= 1e-6

    # Problem hierarchy setup
    #problems = []
    #for i, n in enumerate(n_levels):
    #    setup = SemilinearEllipticSetup(n, alpha, beta)
    #    var = setup.returnVars()
    #    reduced_obj = ReducedObjective(setup)
    #    pde_obj = PDEObjective(setup)
    #    p = SimpleNamespace(
            #setup=setup,
    #        var = var,
    #        obj_smooth=ReducedObjective(setup),  # Your existing wrapper
    #        obj_nonsmooth=L1NormRegularizer(setup),
    #        pvector=L2ControlPrimal(setup),
    #        dvector=L2ControlDual(setup),
            
    #        R=restriction_R_Reduced_2d(n_levels[i+1], n+1) if i < len(n_levels)-1 else np.eye(n_levels[i]),
    #        reduced_obj = reduced_obj
    #     )
    #    problems.append(p)
    # Initial guess (zero control)
    #fine_setup = problems[0].setup
    

    problems = []
    setups = []
    for i, n in enumerate(n_levels):
        setup = SemilinearEllipticSetup(n, alpha, beta)
        setups.append(setup)
        var = setup.returnVars()
        reduced_obj = ReducedObjective(setup)
        nonsmooth_obj = L1NormRegularizer(setup)
        pde_obj = PDEObjective(setup)
        
        p = SimpleNamespace(
            var=var,
            obj_smooth=reduced_obj,
            obj_nonsmooth=L1NormRegularizer(setup),
            pvector=L2ControlPrimal(setup),
            dvector=L2ControlDual(setup),
            R=restriction_R_Reduced_2d(n_levels[i+1], n+1) if i < len(n_levels)-1 else np.eye(n_levels[i]),
        )
        problems.append(p)

# Now use `fine_setup` for plotting, checking residuals etc.
    fine_setup=setups[0]
    import matplotlib.pyplot as plt
    plt.figure()
    plot_control(fine_setup.y_d)
    plt.title("Target State y_d")
    plt.show()

    
    #x0 = np.random.rand(fine_setup.V.dofmap.index_map.size_local)*0.1
    x0 = np.random.rand(fine_setup.V.dofmap.index_map.size_local)*1e-2
    

    # Run optimization
    print("\nStarting multilevel optimization...")
    u_opt, stats = trustregion(0, x0, params['delta'], problems, params)
    def check_state_equation_residual(setup, u_opt, tol=1e-6):
        setup.u.x.array[:] = u_opt
        setup.solve_state_equation()
    
        # Define residual expression (not Form!)
        residual_expr = -ufl.div(ufl.grad(setup.y)) + setup.y**3 - setup.u
    
        # Project residual into a function space
        V = setup.V
        residual_func = fem.Function(V)
        residual_func.interpolate(fem.Expression(residual_expr, V.element.interpolation_points()))
    
        # Compute L² norm
        residual_norm = np.sqrt(fem.assemble_scalar(fem.form(residual_func**2 * ufl.dx)))
        print(f"State equation residual (L² norm): {residual_norm:.3e}")
    
        if residual_norm < tol:
            print("✅ State equation is satisfied.")
        else:
            print("❌ State Residual is too large.")
        return residual_norm

    def check_adjoint_residual(setup, tol=1e-6):
        setup.solve_adjoint_equation()
    
        residual_expr = -ufl.div(ufl.grad(setup.p)) + 3 * setup.y**2 * setup.p + (setup.y - setup.y_d)
    
        V = setup.V
        residual_func = fem.Function(V)
        residual_func.interpolate(fem.Expression(residual_expr, V.element.interpolation_points()))
    
        residual_norm = np.sqrt(fem.assemble_scalar(fem.form(residual_func**2 * ufl.dx)))
        print(f"Adjoint equation residual (L² norm): {residual_norm:.3e}")
    
        if residual_norm < tol:
            print("✅ Adjoint equation is satisfied.")
        else:
            print("❌ Adjoint Residual is too large.")
        return residual_norm

    def check_optimality(setup, u_opt, tol=1e-6):
        """Verify the optimality condition."""
        setup.u.x.array[:] = u_opt
        setup.solve_state_equation()
        setup.solve_adjoint_equation()
        
        # Compute the gradient of the reduced objective
        gradient = setup.alpha * setup.u.x.array + setup.beta * np.sign(setup.u.x.array) + setup.p.x.array
        
        # Check the L² norm of the gradient
        grad_norm = np.linalg.norm(gradient)
        print(f"Optimality condition residual (L² norm): {grad_norm:.3e}")
        
        if grad_norm < tol:
            print("✅ Optimality condition is satisfied.")
        else:
            print("❌ Gradient norm is too large.")
        return grad_norm

        
        
        
    
        
        
    # After optimization (u_opt is the final control)
    res_state = check_state_equation_residual(fine_setup, u_opt)
    res_adjoint = check_adjoint_residual(fine_setup)
    grad_norm = check_optimality(fine_setup, u_opt)
    #plot_residuals(fine_setup, u_opt)  

    # === Results Visualization ===
    plt.figure(figsize=(15, 5))
    
    # 1. Convergence plot
    plt.subplot(1, 3, 1)
    plt.semilogy(stats['objhist'], 'b-o', markersize=4)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.title('Convergence History')
    plt.grid(True)

    # 2. Optimal control
    plt.subplot(1, 3, 2)
    fine_setup.u.x.array[:] = u_opt
    plot_control(fine_setup.u)
    plt.title('Optimal Control')

    # 3. State vs target
    plt.subplot(1, 3, 3)
    fine_setup.solve_state_equation()
    plot_state_vs_target(fine_setup.y, fine_setup.y_d)
    plt.title('State vs Target')

    plt.tight_layout()
    plt.show()

    # === Print final statistics ===
    print("\n=== Optimization Results ===")
    print(f"Final objective: {stats['objhist'][-1]:.3e}")
    print(f"Gradient norm: {stats['gnormhist'][-1]:.3e}")
    print(f"Iterations: {stats['iter']}")
    print(f"Function evals: {stats['nobj1']}")
    print(f"Gradient evals: {stats['ngrad']}")

    return stats

# Visualization helper functions
import matplotlib.pyplot as plt
import numpy as np
from dolfinx.fem import FunctionSpace
from dolfinx.mesh import Mesh
import numpy as np
from dolfinx.mesh import create_rectangle, CellType



def plot_control(u):
    """2D Matplotlib plot that supports triangle or quad meshes."""
    mesh = u.function_space.mesh
    if mesh.topology.dim != 2:
        raise ValueError("Matplotlib can only plot 2D meshes.")

    points = mesh.geometry.x[:, :2]  # x, y coords
    tdim = mesh.topology.dim
    num_vertices = mesh.topology.connectivity(tdim, 0).array.size // mesh.topology.index_map(tdim).size_local

    # Triangle mesh
    if num_vertices == 3:
        cells = mesh.topology.connectivity(tdim, 0).array.reshape(-1, 3)

    # Quad mesh → split into triangles
    elif num_vertices == 4:
        quads = mesh.topology.connectivity(tdim, 0).array.reshape(-1, 4)
        cells = np.vstack([
            quads[:, [0, 1, 2]],
            quads[:, [0, 2, 3]]
        ])
    else:
        raise ValueError(f"Unsupported cell with {num_vertices} vertices")

    plt.tripcolor(points[:, 0], points[:, 1], cells, u.x.array.real, cmap="viridis")
    plt.colorbar(label="Control magnitude")
    plt.show()




def plot_state_vs_target(y, y_d):
    """Compare state and target solutions"""
    mesh = y.function_space.mesh
    plt.plot(y.x.array, 'b-', label='State')
    plt.plot(y_d.x.array, 'r--', label='Target')
    plt.xlabel('DOF index')
    plt.ylabel('Solution value')
    plt.legend() 
    
    
    
    
    
stats = driver_pde(savestats=True, name="semilinear_opt")

# Access detailed history if needed
plt.figure()
plt.plot(stats['deltahist'], label='Trust region radius')
plt.plot(stats['gnormhist'], label='Gradient norm')
plt.legend()   
