import numpy as np
class modelTR:
    def __init__(self, problems, secant, subtype = 'spg', l = 0, R = np.empty(1), grad = np.empty(1), x = np.empty(1)):
        self.problem = problems[l]
        self.var     = problems[l].var
        self.secant  = secant
        self.l       = l
        self.x       = R @ x
        self.R       = R
        self.Rgrad   = problems[l].dvector.dual(R @ grad) #should be in dual space, dgrad in primal
        self.subtype = subtype
        self.nobj1   = 0
        self.ngrad   = 0
        self.nhess   = 0
        if subtype == 'recursive':
            grad, _      = problems[l].obj_smooth.gradient(self.x, 0.)
            self.grad    = problems[l].dvector.dual(grad)
            self.ngrad  += 1

    def update(self, x, type):
        self.problem.obj_smooth.update(x, type)
    def value(self, x, ftol):
        val, ferr    = self.problem.obj_smooth.value(x, ftol)
        if self.subtype == 'recursive':
          val      += self.problem.pvector.dot(self.Rgrad - self.grad, x - self.x)
          # val      += self.problem.dvector.apply(self.Rgrad, x - self.x)
          ferr      = 0
        self.nobj1 += 0
        return val, ferr
    def gradient(self,x,gtol):
      grad, gerr      = self.problem.obj_smooth.gradient(x, gtol)
      if self.subtype == 'recursive':
        grad        += self.problem.pvector.dual(self.Rgrad - self.grad)
        # grad        += self.Rgrad
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

