

import numpy as np
import matplotlib.pyplot as plt



    
class PDEproblem:
    def __init__(self,domain,boundary_conditions,initial_conditions = None,num_points = 50,dimensions =1):
        """
        
        Initialize the PDE problem.

        Parameters
        ----------
        domain : A list of tuples defining the spatial domain in each dimension, e.g. [(x_min,x_max),(y_min,y_max)].
        boundary_conditions : A function or list of functions defining the boundary conditions.
        initial_conditions : A function defining the initial conditions.
        dimensions: The number of spatial dimensions (1,2 or more)

        Returns
        -------
        None.

        """
        self.domain = domain
        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions
        self.num_points = num_points
        self.dimensions = dimensions
        
    @classmethod
    def poisson(cls,domain,boundary_conditions,source_term,num_points = 100, dimensions = 1):
        return PoissonPDE(domain, boundary_conditions, source_term, num_points,dimensions = 1)
    
    @classmethod
    def heat(cls,domain,boundary_conditions,initial_conditions,alpha,num_points = 100, time_steps = 100,dimensions = 1):
        return HeatEquationPDE(domain,boundary_conditions,alpha,num_points,time_steps,dimensions = 1)           
        
    def solve(self):
        raise NotImplementedError("The solve method must be implemented by the subclass.")
        
    def plot_solution(self):
        raise NotImplementedError("The plot_solution mthod must be implemented by the subclass.")
        
        
class TimeDependentPDEProblem:
    def __init__(self,spatial_domain,time_domain,boundary_conditions,initial_conditions,dimensions):
        """
        Initialize the time-dependent PDE problem.

        Parameters
        ----------
        spatial_domain : A list of tuples defining the spatial domain in each dimension, e.g., [(x_min,x_max),(y_min,y_max)].
        time_domain : A tuple defining the time domain, e.g., (t_min,t_max).
        boundary_conditions : A function or list of functions defining the boundary conditions.
        initial_conditions : A function defining the initial conditions.
        dimensions : The number of spatial dimensions (1,2, or more).

        Returns
        -------
        None.

        """
        self.spatial_domain = spatial_domain
        self.time_domain = time_domain
        self.boundary_conditions = boundary_conditions
        self.initial_conditions = initial_conditions
        self.dimensions = dimensions
        
        def setup_grid(self,num_points_spatial,num_points_time):
            """
            Set up the spatial and time grid.

            Parameters
            ----------
            num_points_spatial : A list specifying the number of points in each spatial dimension.
            num_points_time : Number of pointsin the time dimension.

            """
            self.grids = [np.linspace(d[0],d[1],num) for d, num in zip(self.spatial_domain,num_points_spatial)]
            self.meshes = np.meshgrid(*self.grids,indexing='ij')
            self.time_grid = np.linspace(self.time_domain[0],self.time_domain[1],num_points_time)
            
            
        def apply_boundary_conditions(self):
            """
            Apply boundary conditions to the grid.

            """
            #Placeholder for applying boundary conditions, should be adapted to the problem
            pass
    
        def apply_initial_conditions(self):
            """
            Apply initial conditions to the grid

            """
            if self.dimensions == 1:
                self.solution = self.initial_conditions(self.meshes[0])
            elif self.dimensions == 2:
                self.solution = self.initial_conditions(*self.meshes)
            else:
                self.solution = self.initial_conditions(*self.meshes)
                
        def solve(self,time_step,method = 'forward_euler'):
            """
            Solve the time-dependent PDE using a specified time-stepping method.

            Parameters
            ----------
            time_step : Time step size for the time-stepping method.
            method : The time-stepping method to use ('forward_euler','backward_euler',etc.). The default is 'forward_euler'.

            """
            num_steps = len(self.time_grid)
            dx = self.grids[0][1]-self.grids[0][0] #Assuming uniform grid in each dimension
            
            for t in range(1,num_steps):
                u_new = np.copy(self.solution)
                if self.dimensions == 1:
                    u_new[1:-1] = self.solution[1:-1]+self.alpha*time_step/dx**2 * (
                        self.solution[:-2] - 2*self.solution[1:-1] +self.solution[2:]
                        )
                elif self.dimensions ==2:
                    u_new[1:-1,1:-1] = self.solution[1:-1,1:-1]+self.alpha*time_step/dx**2 *(
                        self.solution[:-2,1:-1] - 2*self.solution[1:-1,1:-1]+self.solution[2:,1:-1]
                        ) +self.alpha*time_step/dx**2 *(
                            self.solution[1:-1,:-2]-2*self.solution[1:-1,1:-1]+self.solution[1:-1.2:])
                #Add other dimensions if needed
                self.apply_boundary_conditions()
                self.solution = u_new
                
        def plot_solution(self,time_index = -1):
            """
            Plot the solution to the PDE at a specific time step.
            
            Parameters
            ----------
            time_index : Index of the time step to plot. Default is the last time step.
            """
            
            if self.dimensions == 1:
                plt.plot(self.grids[0],self.solution)
                plt.title(f"PDE Solution in 1D at time {self.time_grid[time_index]:.2f}")
                plt.xlabel("X")
                plt.ylabel("Solution")
            elif self.dimensions == 2:
                plt.plot(self.grids[0],self.solution)
                plt.title(f"PDE Solution in 2D at time {self.time_grid[time_index]:.2f}")
                plt.xlabel("X")
                plt.ylabel("Y")
                
            plt.show()
            
           

class PoissonPDE:
    def __init__(self, domain, boundary_conditions, source_term, num_points, dimensions=1):
        self.domain = domain  # [(x_min, x_max)] for 1D, [(x_min, x_max), (y_min, y_max)] for 2D
        self.bc = boundary_conditions  # Boundary condition function or list of values
        self.f = source_term  # Source term function f(x) or f(x, y)
        self.num_points = num_points  # Number of grid points (int for 1D, tuple/list for 2D)
        self.dimensions = dimensions  # 1 or 2 for 1D or 2D
        
        if dimensions == 1:
            self.h = (domain[1] - domain[0]) / (num_points - 1)  # Grid spacing in 1D
            self.x = np.linspace(domain[0], domain[1], num_points)  # Discretized domain in 1D
            self.solution = np.zeros(num_points)  # Solution vector in 1D
        elif dimensions == 2:
            self.hx = (domain[0][1] - domain[0][0]) / (num_points[0] - 1)  # Grid spacing in x
            self.hy = (domain[1][1] - domain[1][0]) / (num_points[1] - 1)  # Grid spacing in y
            self.x = np.linspace(domain[0][0], domain[0][1], num_points[0])  # Discretized x domain
            self.y = np.linspace(domain[1][0], domain[1][1], num_points[1])  # Discretized y domain
            self.solution = np.zeros((num_points[0], num_points[1]))  # Solution matrix in 2D

    def solve(self, max_iter=5000):
        if self.dimensions == 1:
            return self.solve_1d()
        elif self.dimensions == 2:
            return self.solve_2d(max_iter)
        
    def apply_boundary_conditions(self):
        if self.dimensions == 1:
            return self.apply_boundary_conditions_1d()
        elif self.dimensions == 2:
            return self.apply_boundary_conditions_2d()

    def solve_1d(self):
        # Create the matrix A for 1D
        A = np.diag(-2 * np.ones(self.num_points)) + np.diag(np.ones(self.num_points-1), 1) + np.diag(np.ones(self.num_points-1), -1)
        A[0, :] = A[-1, :] = 0
        A[0, 0] = A[-1, -1] = 1

        # Create the right-hand side vector
        b = self.f(self.x) * self.h**2
        b[0] = self.bc[0]
        b[-1] = self.bc[-1]

        # Solve the linear system
        self.solution = np.linalg.solve(A, b)
        return self.solution
    
    def apply_boundary_conditions_1d(self):
        # Apply boundary conditions in 2D
        self.solution[0] = self.bc[0]
        self.solution[-1] = self.bc[-1]

    def solve_2d(self, max_iter):
        # Right-hand side matrix for 2D
        b = self.f(self.x[:, None], self.y[None, :]) * self.hx**2

        # Iterative solver (Jacobi method)
        for _ in range(max_iter):
            self.solution[1:-1, 1:-1] = 0.25 * (
                self.solution[:-2, 1:-1] + self.solution[2:, 1:-1] +
                self.solution[1:-1, :-2] + self.solution[1:-1, 2:] - b[1:-1, 1:-1]
            )
            self.apply_boundary_conditions_2d()

        return self.solution

    def apply_boundary_conditions_2d(self):
        # Apply boundary conditions in 2D
        self.solution[:, 0] = self.bc(self.x)
        self.solution[:, -1] = self.bc(self.x)
        self.solution[0, :] = self.bc(self.y)
        self.solution[-1, :] = self.bc(self.y)

    def plot_solution(self):
        if self.dimensions == 1:
            plt.plot(self.x, self.solution, label="Numerical Solution")
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.title("1D Poisson Equation Solution")
            plt.legend()
            plt.show()
        elif self.dimensions == 2:
            X, Y = np.meshgrid(self.x, self.y)
            plt.contourf(X, Y, self.solution, cmap="viridis")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("2D Poisson Equation Solution")
            plt.show()

# Example usage for 1D:
def quadratic(x,a = -1.0):
    return 0.5*a*x**2

def constant_1(x):
    return -1.0+x-x

domain_1d = (0, 1)
boundary_conditions_1d = [0, -0.5]
num_points_1d = 100

pde_1d = PoissonPDE(domain_1d, boundary_conditions_1d, constant_1, num_points_1d, dimensions=1)
solution_1d = pde_1d.solve()
pde_1d.plot_solution()


# Example usage for 2D:
def source_term_2d(x, y):
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def boundary_conditions_2d(x):
    return np.zeros_like(x)

domain_2d = [(0, 1), (0, 1)]
num_points_2d = [50, 50]

pde_2d = PoissonPDE(domain_2d, boundary_conditions_2d, source_term_2d, num_points_2d, dimensions=2)
solution_2d = pde_2d.solve()
pde_2d.plot_solution()







class HeatEquationPDE:
    def __init__(self, domain, boundary_conditions, initial_conditions, alpha, num_points=100, time_steps=100, dimensions=1):
        self.domain = domain
        self.bc = boundary_conditions
        self.ic = initial_conditions
        self.alpha = alpha
        self.num_points = num_points
        self.time_steps = time_steps
        self.dimensions = dimensions

        if dimensions == 1:
            self.h = (domain[1] - domain[0]) / (num_points - 1)
            self.x = np.linspace(domain[0], domain[1], num_points)
            self.solution = np.zeros((time_steps, num_points))
            self.solution[0, :] = initial_conditions(self.x)
        elif dimensions == 2:
            self.hx = (domain[0][1] - domain[0][0]) / (num_points - 1)
            self.hy = (domain[1][1] - domain[1][0]) / (num_points - 1)
            self.x = np.linspace(domain[0][0], domain[0][1], num_points)
            self.y = np.linspace(domain[1][0], domain[1][1], num_points)
            self.solution = np.zeros((time_steps, num_points, num_points))
            X, Y = np.meshgrid(self.x, self.y)
            self.solution[0, :, :] = initial_conditions(X, Y)
        else:
            raise ValueError("Only 1D and 2D cases are supported.")

    def apply_boundary_conditions(self, u):
        if self.dimensions == 1:
            u[0] = self.bc[0](self.x[0])
            u[-1] = self.bc[1](self.x[-1])
        elif self.dimensions == 2:
            u[0, :] = self.bc[0](self.x)
            u[-1, :] = self.bc[1](self.x)
            u[:, 0] = self.bc[2](self.y)
            u[:, -1] = self.bc[3](self.y)
        return u

    def solve(self):
        dt = 1.0 / self.time_steps
        if self.dimensions == 1:
            r = self.alpha * dt / self.h**2
            for n in range(0, self.time_steps - 1):
                self.solution[n+1, 1:-1] = (
                    self.solution[n, 1:-1]
                    + r * (self.solution[n, 2:] - 2*self.solution[n, 1:-1] + self.solution[n, :-2])
                )
                self.solution[n+1, :] = self.apply_boundary_conditions(self.solution[n+1, :])
        elif self.dimensions == 2:
            rx = self.alpha * dt / self.hx**2
            ry = self.alpha * dt / self.hy**2
            for n in range(0, self.time_steps - 1):
                self.solution[n+1, 1:-1, 1:-1] = (
                    self.solution[n, 1:-1, 1:-1]
                    + rx * (self.solution[n, 2:, 1:-1] - 2*self.solution[n, 1:-1, 1:-1] + self.solution[n, :-2, 1:-1])
                    + ry * (self.solution[n, 1:-1, 2:] - 2*self.solution[n, 1:-1, 1:-1] + self.solution[n, 1:-1, :-2])
                )
                self.solution[n+1, :, :] = self.apply_boundary_conditions(self.solution[n+1, :, :])
        return self.solution

    def plot_solution(self):
        if self.dimensions == 1:
            plt.plot(self.x, self.solution[-1, :], label="Numerical Solution")
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.title("1D Heat Equation Solution")
            plt.legend()
            plt.show()
        elif self.dimensions == 2:
            X, Y = np.meshgrid(self.x, self.y)
            plt.contourf(X, Y, self.solution[-1, :, :], cmap="viridis")
            plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("2D Heat Equation Solution")
            plt.show()


##Test for heat equation pde
# Define boundary conditions, initial conditions, and parameters for 1D
def initial_conditions_1d(x):
    return np.sin(np.pi * x)

pde_heat_1d = HeatEquationPDE(
    domain=(0, 1),
    boundary_conditions=[lambda x: 0, lambda x: 0],
    initial_conditions=initial_conditions_1d,
    alpha=0.01,
    num_points=100,
    time_steps=100,
    dimensions=1
)

solution_1d = pde_heat_1d.solve()
pde_heat_1d.plot_solution()



# Define boundary conditions, initial conditions, and parameters for 2D
def initial_conditions_2d(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

pde_heat_2d = HeatEquationPDE(
    domain=((0, 1), (0, 1)),
    boundary_conditions=[
        lambda x: 0,  # Boundary at y=0
        lambda x: 0,  # Boundary at y=1
        lambda y: 0,  # Boundary at x=0
        lambda y: 0   # Boundary at x=1
    ],
    initial_conditions=initial_conditions_2d,
    alpha=0.01,
    num_points=50,
    time_steps=100,
    dimensions=2
)

solution_2d = pde_heat_2d.solve()
pde_heat_2d.plot_solution()
My_problem = PDEproblem.poisson(domain_1d, boundary_conditions_1d, constant_1)
print(My_problem)
