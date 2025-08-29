import numpy as np

class modelTR:
    def __init__(self, problems, secant, subtype = 'spg', l = 0, R = np.empty(1), dgrad = np.empty(1), x = np.empty(1)):
        self.problem = problems[l]
        self.var     = problems[l].var
        self.secant  = secant
        self.l       = l
        self.x       = x
        self.R       = R
        # self.Rgrad   = problems[l].dvector.dual(R @ grad) #should be in dual space, dgrad in primal
        self.Rdgrad   = dgrad #store old grad and dual grad
        self.Rgrad    = problems[l].pvector.dual(self.Rdgrad)
        self.subtype  = subtype
        self.nobj1    = 0
        self.ngrad    = 0
        self.nhess    = 0
        if subtype == 'recursive': #store grad at x
            self.problem.obj_smooth.update(self.x, 'init')
            self.grad, _      = self.problem.obj_smooth.gradient(self.x, 1.e-6)
            self.dgrad        = self.problem.dvector.dual(self.grad)
            self.ngrad       += 1
    def update(self, x, type):
        self.problem.obj_smooth.update(x, type)
    def value(self, x, ftol):
        val, ferr    = self.problem.obj_smooth.value(x, ftol)
        if self.subtype == 'recursive':
          val      += self.problem.pvector.dot(self.Rdgrad - self.dgrad, x - self.x)
          ferr      = 0
        self.nobj1 += 0
        return val, ferr
    def gradient(self,x,gtol):
      grad, gerr      = self.problem.obj_smooth.gradient(x, gtol)
      if self.subtype == 'recursive':
        # import pdb
        # pdb.set_trace()
        grad        += self.Rgrad - self.grad
        self.ngrad  += 0
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

