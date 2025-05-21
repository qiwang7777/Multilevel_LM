#Problem Class
from .L2Vectors import Euclidean, L2vectorPrimal, L2vectorDual

class Problem:
    def __init__(self, var, R):
        self.var  = var
        if var['useEuclidean']:
            self.pvector   = Euclidean(var)
            self.dvector   = Euclidean(var)
        else:
            self.pvector   = L2vectorPrimal(var)
            self.dvector   = L2vectorDual(var)
        # self.obj_smooth    = ReducedObjective(Objective(var), ConstraintSolver(var))
        # self.obj_nonsmooth = L1Norm(var)
        self.R             = R