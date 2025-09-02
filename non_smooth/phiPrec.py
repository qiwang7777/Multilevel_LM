import numpy as np
class phiPrec: # you can definitely clean this up and inherit a bunch of stuff but
               # we can be explicit for now
    def __init__(self, problem, R = np.empty(1), l = 0, x = np.empty(1), t = 1):
        self.problem   = problem
        self.var       = problem.obj_nonsmooth.var
        self.R         = R
        self.l         = l
        self.nobj2     = 0
        self.nprox     = 0
        self.xinit     = x # '' fine prox for first iteration ''

    def value(self, x):
        if x.shape[0] == self.xinit.shape[0]:
             c1 = np.linalg.norm(x - self.xinit)
        else:
            c1 = 1e-10
        if self.l == 0 or c1 < 1e-8:
            val             = self.problem.obj_nonsmooth.value(x)
        else:
            val             = self.problem.obj_nonsmooth.value(self.R.T @ x)

        self.nobj2     += 1
        return val
    def prox(self, x, t):
        if x.shape[0] == self.xinit.shape[0]:
            c1 = np.linalg.norm(x - self.xinit)
        else:
            c1 = 1e-10
        if self.l == 0 or c1 < 1e-8:
            px          = self.problem.obj_nonsmooth.prox(x, t)
        else:
            px          = self.R @ self.problem.obj_nonsmooth.prox(self.R.T @ x, t) # x = x_{i-1,j} + alpha * s_i
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