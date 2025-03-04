import scipy.sparse as sp
import numpy as np

class ConstraintSolver:
    def __init__(self, var):
        
        self.var = var
        self.uprev = np.ones(self.var['nu'])
        
    def begin_counter(self,iter,cnt0):
        return cnt0
    
    def end_counter(self,iter,cnt0):
        return cnt0
        

    def solve(self, z, stol=1e-12):
        nu = self.var['nu']
        u = self.uprev
        c, _ = self.value(np.hstack([u,z]))
        cnt = 0 
        atol = stol
        rtol = 1
        cnorm = np.linalg.norm(c)
        ctol = min(atol, rtol * cnorm)
        
        for _ in range(100):
            s,_ = self.apply_inverse_jacobian_1(self.value(np.hstack([u, z]))[0], np.hstack([u, z]))
            
            unew = u - s
            cnew = self.value(np.hstack([unew, z]))[0]
            ctmp = np.linalg.norm(cnew)
            
            alpha = 1
            while ctmp > (1 - 1e-4 * alpha) * cnorm:
                alpha *= 0.1
                unew = u - alpha * s
                cnew = self.value(np.hstack([unew, z]))[0]
                ctmp = np.linalg.norm(cnew)
            
            u, c, cnorm = unew, cnew, ctmp
            cnt += 1
            if cnorm < ctol:
                break
        serr = cnorm
        
        self.uprev = u
        return u,cnt,serr

    def reset(self):
        self.uprev = np.ones(self.var['nu'])

    def value(self,x):
        nu = self.var['nu']
        A, B, b = self.var['A'], self.var['B'], self.var['b']
        u, z = x[:nu], x[nu:]
        Nu = self.evaluate_nonlinearity_value(u)
        #print("A_shape:", A.shape) #(n-1,n-1)
        #print("u_shape:",u.shape)  #(n-1,)
        #print("Nu_shape:", Nu.shape) #(n-1,)
        #print("B_shape:", B.shape) #(n-1,n)
        #print("z_shape:",z.shape) #(n,)
        #print("b_shape:",b.shape) #(n-1,)
        c = A @ u + Nu - (B @ z + b)
        return c, 0

    def apply_jacobian_1(self, v, x, gtol=1e-6):
        nu = self.var['nu']
        A = self.var['A']
        u = x[:nu]
        
        J = self.evaluate_nonlinearity_jacobian(u)
        return (A + J) @ v, 0

    def apply_jacobian_2(self, v,x,gtol=1e-6):
        return -self.var['B'] @ v, 0

    def apply_adjoint_jacobian_1(self, v, x):
        nu = self.var['nu']
        A = self.var['A']
        u = x[:nu]
        J = self.evaluate_nonlinearity_jacobian(u)
        return (A + J).T @ v, 0

    def apply_adjoint_jacobian_2(self, v,x,gtol=1e-6):
        return -self.var['B'].T @ v, 0

    def apply_inverse_jacobian_1(self, v, x,gtol=1e-6):
        nu = self.var['nu']
        A = self.var['A']
        u = x[:nu]
        
        J = self.evaluate_nonlinearity_jacobian(u)
        
        solution = sp.linalg.spsolve(A+J,v)
        return solution, 0

    def apply_inverse_adjoint_jacobian_1(self, v, x,gtol=1e-6):
        nu = self.var['nu']
        A = self.var['A']
        u = x[:nu]
        J = self.evaluate_nonlinearity_jacobian(u)
        return sp.linalg.spsolve((A + J).T, v), 0

    def apply_adjoint_hessian_11(self, u, v, x,htol=1e-6):
        nu = self.var['nu']
        ahuv = np.zeros(nu)
        for i in range(nu):
            if i < nu - 1:
                ahuv[i] += (u[i] * v[i + 1] - u[i + 1] * (2 * v[i] + v[i + 1])) / 6
            if i > 0:
                ahuv[i] += (u[i - 1] * (v[i - 1] + 2 * v[i]) - u[i] * v[i - 1]) / 6
        return ahuv, 0

    def apply_adjoint_hessian_12(self, u, v, x):
        return np.zeros(self.var['nz']), 0

    def apply_adjoint_hessian_21(self, u, v, x):
        return np.zeros(self.var['nu']), 0

    def apply_adjoint_hessian_22(self, u, v, x):
        return np.zeros(self.var['nz']), 0

    def evaluate_nonlinearity_value(self, u):
        n = self.var['n']
        Nu = np.zeros(n - 1)
        Nu[:-1] += u[:-1] * u[1:] + u[1:] ** 2
        Nu[1:] -= u[:-1] * u[1:] + u[:-1] ** 2
        return Nu / 6

    def evaluate_nonlinearity_jacobian(self, u):
        n = self.var['n']
        
        
       
        d1, d2, d3 = np.zeros(n - 1), np.zeros(n - 1), np.zeros(n - 1)
       
        d1[:-1] = u[:-1] - u[1:]
        d2[0] = u[1]
        d2[1:-1] = u[2:]-u[:-2]
        d2[-1] =  -u[-2]
        d3[1:] = u[:-1] + 2 * u[1:]
        J = spdiags([d1, d2, d3], [-1, 0, 1], n-1, n-1) / 6
        
        return J

    def profile(self):
        return []
