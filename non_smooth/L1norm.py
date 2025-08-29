import torch
import numpy as np
class L1TorchNorm:
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
    def subdiff(self,x):
        subdiff_result = {}
        for k,v in x.td.items():
            subdiff_result[k] = torch.where(
                v>0, torch.tensor(1.0),
                torch.where(v<0,torch.tensor(-1.0),torch.tensor([-1.0,1.0]))
            )


        return subdiff_result

    def get_parameter(self):
        return self.var['beta']


class L1Norm:
    def __init__(self, var):
        self.var = var

    def value(self, x):
        return self.var['beta'] * np.dot(self.var['Rlump'].T, np.abs(x))

    def prox(self, x, t):
        if self.var['useEuclidean']:
            return np.maximum(0, np.abs(x) - t * self.var['Rlump'] * self.var['beta']) * np.sign(x)
        else:
            return np.maximum(0, np.abs(x) - t * self.var['beta']) * np.sign(x)

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

class L1NormEuclid:
    def __init__(self, var):
        self.var = var

    def value(self, x):
        return self.var.beta * np.dot(self.var.R.T, np.abs(x))

    def prox(self, x, t):
        return np.maximum(0, np.abs(x) - t * self.var.R * self.var.beta) * np.sign(x)

    def dir_deriv(self, s, x):
        sx = np.sign(x)
        return self.var.beta * (np.dot(sx.T, s) + np.dot((1 - np.abs(sx)).T, np.abs(s)))

    def project_sub_diff(self, g, x):
        sx = np.sign(x)
        return self.var.beta * sx + (1 - np.abs(sx)) * np.clip(g, -self.var.beta, self.var.beta)

    def gen_jac_prox(self, x, t):
        d = np.ones_like(x)
        px = self.prox(x, t)
        ind = px == 0
        d[ind] = 0
        return np.diag(d), ind

    def apply_prox_jacobian(self, v, x, t):
        ind = np.abs(x) <= t * self.var.M0 * self.var.beta
        Dv = v.copy()
        Dv[ind] = 0
        return Dv

    def get_parameter(self):
        return self.var.beta