import copy, torch
class TorchVect:
    @torch.no_grad()
    def __init__(self, tensordict): #store tensor dictionary
        self.td = tensordict
        s = []
        for _, v in self.td.items():
            s.append(v.size()[0])
        self.shape = s
    @torch.no_grad()
    def clone(self):
        td  = copy.deepcopy(self.td)
        ans = TorchVect(td)
        ans.zero()
        return ans
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
    def __matmul__(self, R):
        ans = self.clone()
        for k,v in self.td.items():
            print(k, v.size(), R.td[k].size())
            ans.td[k] = R.td[k] @ v
            # if k == 'fc1.weight':
            #     ans.td[k] = R @ v
            # elif k == 'fc1.bias':
            #     ans.td[k] = R @ v
            # elif k =='fc2.weight':
            #     ans.td[k] = R @ v @ R.T
            # elif k =='fc2.bias':
            #     ans.td[k] = R @ v
            # elif k == 'fc3.weight':
            #     ans.td[k] = v @ R.T
            # else:
            #     ans.td[k] = v
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