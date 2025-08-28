import torch
import copy
class L1TorchNorm:
    def __init__(self, var):
        # expects: var["beta"] (float), optional var["l1_exclude"] = ('bias','cB')
        self.var = var
        self.beta = float(var.get('beta', 0.0))
        self.exclude = tuple(var.get('l1_exclude', ('bias','cB')))

    def _items(self, x):
        td = x.td if hasattr(x, "td") else x
        return td.items(), td

    def value(self, x):
        total = 0.0
        items, _ = self._items(x)
        for k, v in items:
            if not torch.is_tensor(v): 
                continue
            if any(ex in k for ex in self.exclude):
                continue
            total += float(v.abs().sum().item())
        return self.beta * total

    def prox(self, x, t):
        """Soft-threshold each parameter (except excluded). Prox of beta*||·||_1."""
        lam = self.beta * float(t)
        out = x.clone() if hasattr(x, "clone") else copy.deepcopy(x)
        items_in, td_in = self._items(x)
        items_out, td_out = self._items(out)
        for k, v in td_in.items():
            if not torch.is_tensor(v):
                td_out[k] = v
                continue
            if any(ex in k for ex in self.exclude):
                td_out[k] = v.clone()
                continue
            td_out[k] = torch.sign(v) * torch.clamp(v.abs() - lam, min=0.0)
        return out

    # Optional helpers; only keep if you actually use them
    def dir_deriv(self, s, x):
        total = 0.0
        xs, _ = self._items(x)
        ss, _ = self._items(s)
        for k, xk in xs:
            sk = (s.td if hasattr(s,"td") else s)[k]
            sign_x = torch.sign(xk)
            # subgradient directional derivative (one selection)
            total += float((sign_x * sk + (1.0 - sign_x.abs()) * sk.abs()).sum().item())
        return self.beta * total

    def project_sub_diff(self, g, x):
        # projection of g onto ∂(beta*||x||_1)
        proj = {}
        td_x = x.td if hasattr(x,"td") else x
        td_g = g.td if hasattr(g,"td") else g
        for k in td_x.keys():
            xk, gk = td_x[k], td_g[k]
            s = torch.sign(xk)
            proj[k] = self.beta * s + (1 - s.abs()) * torch.clamp(gk, -self.beta, self.beta)
        res = x.clone() if hasattr(x,"clone") else copy.deepcopy(x)
        res.td = proj if hasattr(res,"td") else proj
        return res

    def gen_jac_prox(self, x, t):
        # diagonal of prox Jacobian (1 if |x|>beta*t else 0)
        lam = self.beta * float(t)
        mask = {}
        td = x.td if hasattr(x,"td") else x
        for k, v in td.items():
            if not torch.is_tensor(v) or any(ex in k for ex in self.exclude):
                mask[k] = torch.ones_like(v)
            else:
                mask[k] = (v.abs() > lam).to(v.dtype)
        return mask

    def apply_prox_jacobian(self, v, x, t):
        lam = self.beta * float(t)
        out = v.clone() if hasattr(v,"clone") else copy.deepcopy(v)
        td_v = v.td if hasattr(v,"td") else v
        td_x = x.td if hasattr(x,"td") else x
        td_out = out.td if hasattr(out,"td") else out
        for k in td_v.keys():
            vk = td_v[k]; xk = td_x[k]
            if not torch.is_tensor(vk) or any(ex in k for ex in self.exclude):
                td_out[k] = vk.clone()
            else:
                td_out[k] = vk * (xk.abs() > lam).to(vk.dtype)
        return out

    def get_parameter(self):
        return self.beta

