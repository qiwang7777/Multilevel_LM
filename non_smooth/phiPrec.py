import numpy as np
class phiPrec:
    """
    
    Parameters
    ----------
    problem: the previous level {i-1} Problem.
    R: ndarray of shape (n_i,n_{i-1}), Restriction matrix from level i-1 (fine) to level i (coarse)
    l: int, current level index i >= 0
    x_fine : ndarray | None , The special x_{i-1,k} (on level i-1) to define x_{i,0} = R @ x_fine
    x_tol: float, Tolerance for detecting x = x_{i,0}.
    assume_tight_rows: bool, If True, enable the closed-form prox for the composition branch.
    
    """
    def __init__(self, problems, R:np.ndarray,l: int, x_fine:Optional[np.ndarray]=None, x_tol:float=1e-12, assume_tight_rows:bool=True, use_anchor_in_prox:bool=False):
        self.problems = problems
        self.l = l
        self.R = None if self.l == 0 else R
        self.nobj2 = 0
        self.nprox = 0
        self.x_fine = None if x_fine is None else x_fine
        self.x_tol = float(x_tol)
        self.assume_tight_rows = bool(assume_tight_rows)
        self._x_coarse = None
        self.use_anchor_in_prox = bool(use_anchor_in_prox)
        if self.l>0 and self.R is not None and self.x_fine is not None:
            self._x_coarse = self.R @ self.x_fine
            
    def update_x(self,x_fine:np.ndarray) -> None:
        """
        Update the special fine point x_{i-1,k} and cache x_{i,0} = R @ x_fine

        """
        self.x_fine = x_fine
        if self.l>0 and self.R is not None:
            self._x_coarse = self.R @self.x_fine
        
    def _is_anchor(self,x:np.ndarray) -> bool:
        if self._x_coarse is None:
            return False
        if x.shape != self._x_coarse.shape:
            return False
        return np.allclose(x,self._x_coarse,atol = self.x_tol,rtol=0.0)
    
    def value(self,x:np.ndarray) -> float:
        """
       φ_{c}(x) = φ_{f}(x_f+R^T(x-x_c)) with x_c = Rx_f
     
        """
        if self.l == 0 or self.R is None:
            val = self.problems[0].obj_nonsmooth.value(x)
        else:
            if self._is_anchor(x):
                val = self.problems[self.l-1].obj_nonsmooth.value(self.x_fine)
            else:
                prev = self.problems[self.l-1].obj_nonsmooth
                x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
                y = self.x_fine +self.R.T @ (x-x_coarse)
                val = prev.value(y)
        self.nobj2 += 1
        return val
    def prox(self, u:np.ndarray,t:float) -> np.ndarray:
        """
        prox_{t φ_i}(u) with φ_i(x)=φ_{i-1}( x_fine + R^T (x - x_coarse) )
        Closed-form under RR^T=I:
          let v = u - x_coarse and y = x_fine + R^T v (fine-space),
          then
            z = v + R ( prox_{t φ_{i-1}}(y) - y )
            x = x_coarse + z

        """
        if not self.assume_tight_rows:
            raise NotImplementedError("Closed-form prox requires R R.T=I. Provided a numerical solver otherwise.")
        if self.l == 0 or self.R is None:
            px = self.problems[0].obj_nonsmooth.prox(u,t)
            self.nprox += 1
            return px
        
       
        else:
            
            n_coarse, n_fine = self.R.shape
            if u.shape[0] != n_coarse:
                 raise ValueError(
                    f"phiPrec(prox): level {self.l}: got u.shape[0]={u.shape[0]} but R is {n_coarse}×{n_fine}. "
                    f"u must be the **coarse** vector of length {n_coarse}. "
                    f"Likely the caller passed a fine-level vector or attached phiPrec to the wrong problem."
                )
        #else:
        #    if self._is_anchor(u):
        #        px = self.R @ self.problems[self.l-1].obj_nonsmooth.prox(self.x_fine, t)
        #    else:
        #        px = self.R @ self.problems[self.l-1].obj_nonsmooth.prox(self.R.T @ u ,t)
            prev = self.problems[self.l-1].obj_nonsmooth
            x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
            v = u - x_coarse
            y = self.x_fine +self.R.T @ v
            py = prev.prox(y,t)
            z = v + self.R @(py-y)
            xcomp = x_coarse+z
            if not self.use_anchor_in_prox or self._x_coarse is None:
                self.nprox +=1
                return xcomp
            Fcomp = 0.5 / t * np.linalg.norm(xcomp-u)**2 +prev.value(self.x_fine+self.R.T@(xcomp-x_coarse))
            Fanc = 0.5 / t * np.linalg.norm(self._x_coarse-u)**2+prev.value(self.x_fine)
            px = self._x_coarse if (Fanc +1e-12<Fcomp) else xcomp
            
            self.nprox += 1
            return px
    
    def addCounter(self, cnt):
        cnt["nobj2"] += self.nobj2
        cnt["nprox"] += self.nprox
        return cnt
    
    def genJacProx(self, x, t):
        if self.l == 0 or self.R is None:
            return self.problems[0].obj_nonsmooth.gen_jac_prox(x, t)
        prev = self.problems[self.l-1].obj_nonsmooth
        x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
        y = self.x_fine + self.R.T @ (x - x_coarse)
        D_prev, ind_prev = prev.gen_jac_prox(y, t)
        if D_prev is None: return None, None
        D_prev = np.asarray(D_prev)
        n_i = self.R.shape[0]
        J = np.eye(n_i) + self.R @ (D_prev - np.eye(D_prev.shape[0])) @ self.R.T
        ind = None
        if ind_prev is not None:
            ind = (np.abs(self.R @ ind_prev.astype(float)) > 0).ravel()
        return J, ind
    
    
    def applyProxJacobian(self, v, x, t):
        if self.l == 0 or self.R is None:
            return self.problems[0].obj_nonsmooth.apply_prox_jacobian(v,x,t)
        x_coarse = self._x_coarse if self._x_coarse is not None else (self.R @ self.x_fine)
        y = self.x_fine + self.R.T @ (x - x_coarse)
        Jprev_v = self.problems[self.l-1].obj_nonsmooth.apply_prox_jacobian(self.R.T @ v, y, t)
        #Dv = self.problems[self.l].obj_nonsmooth.applyProxJacobian(v, x, t)
        return v + self.R@(Jprev_v-self.R.T @ v)
    
    def getParameter(self):
        #return self.problem.obj_nonsmooth.getParameter()
        return (self.problems[0].obj_nonsmooth.get_parameter() 
                if self.l==0 else self.problems[self.l-1].obj_nonsmooth.get_parameter())
