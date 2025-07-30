import copy, torch
class TorchVect:
    @torch.no_grad()
    def __init__(self, tensordict, isRop = False): #store tensor dictionary
        self.td    = tensordict
        s          = []
        self.isRop = isRop
        for _, v in self.td.items():
            s.append(v.size()[0])
        self.shape = s
    @torch.no_grad()
    def clone(self):
        td  = copy.deepcopy(self.td)
        ans = TorchVect(td)
        ans.zero()
        return ans

    @property
    def T(self):
        J = len(self.td.items())
        Temp = self.clone()
        for j, (k, v) in enumerate(list(self.td.items())):
              Temp.td[k] = v.T
        return Temp

    @torch.no_grad()
    def zero(self):
        for _, v in self.td.items():
            v.zero_()
    @torch.no_grad()
    def __add__(self, other):
        temp = other.clone()
        for k, v in self.td.items():
            temp.td[k] = other.td[k] + v
        return temp
    @torch.no_grad()
    def __sub__(self, other):
        temp = other.clone()
        for k, v, in self.td.items():
           temp.td[k] = other.td[k] - v
        return -1*temp
    @torch.no_grad()
    def __mul__(self, alpha):
      ans = self.clone()
      for k, v in self.td.items():
          ans.td[k].add_(v, alpha = alpha)
      return ans
    @torch.no_grad()
    def __rmul__(self, alpha):
        return self.__mul__(alpha)
    @torch.no_grad()
    def __matmul__(self, x):
        ans = x.clone()
        I = list(x.td.items())
        for index, (k, v) in enumerate(I):
            import pdb
            pdb.set_trace()
            if index % 2 == 0 and not x.isRop:
              nk = I[index+1][0]
              # print(nk, self.td[nk].size())
              ans.td[k] = self.td[nk] @ (v @ self.td[k])
            else:
              if index == 0 and x.isRop:
                ans.td[k] =  v @ self.td[k]
              else:
                ans.td[k] = self.td[k] @ v
        return ans

    @torch.no_grad()
    def __truediv__(self, alpha):
        ans = self.clone()
        for k, v in self.td.items():
            ans.td[k].add_(v, alpha = 1/alpha)
        return ans
    @torch.no_grad()
    def __rtruediv__(self, alpha):
        return self.__truediv__(alpha)