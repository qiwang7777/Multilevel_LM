#Reduced Objective Function

class ReducedObjective:
    def __init__(self, obj0, con0):
        self.obj0 = obj0
        self.con0 = con0
        self.is_state_computed = False
        self.is_state_cached = False
        self.is_adjoint_computed = False
        self.is_adjoint_cached = False
        self.uwork = None
        self.ucache = None
        self.pwork = None
        self.pcache = None
        self.cnt = {
            "nstate": 0, "nadjoint": 0, "nstatesens": 0,
            "nadjointsens": 0, "ninvJ1": 0
        }
    
    def begin_counter(self, iter, cnt0):
        if iter == 0:
            cnt0.update({
                "nstatehist": [], "nadjoihist": [], "nstsenhist": [],
                "nadsenhist": [], "ninvJ1hist": []
            })
            for key in self.cnt:
                self.cnt[key] = 0
        return self.con0.begin_counter(iter, cnt0)
    
    def end_counter(self, iter, cnt0):
        cnt0["nstatehist"].append(self.cnt["nstate"])
        cnt0["nadjoihist"].append(self.cnt["nadjoint"])
        cnt0["nstsenhist"].append(self.cnt["nstatesens"])
        cnt0["nadsenhist"].append(self.cnt["nadjointsens"])
        cnt0["ninvJ1hist"].append(self.cnt["ninvJ1"])
        return self.con0.end_counter(iter, cnt0)
    
    def reset(self):
        self.uwork = None
        self.ucache = None
        self.is_state_computed = False
        self.is_state_cached = False
        self.pwork = None
        self.pcache = None
        self.is_adjoint_computed = False
        self.is_adjoint_cached = False
    
    def update(self, x, type):
        if type == "init":
            self.is_state_computed = False
            self.is_state_cached = False
            self.is_adjoint_computed = False
            self.is_adjoint_cached = False
        elif type == "trial":
            self.is_state_cached = self.is_state_computed
            self.is_adjoint_cached = self.is_adjoint_computed
            self.is_state_computed = False
            self.is_adjoint_computed = False
        elif type == "reject":
            if self.is_state_cached:
                self.uwork = self.ucache
                self.is_state_computed = True
            else:
                self.is_state_computed = False
            if self.is_adjoint_cached:
                self.pwork = self.pcache
                self.is_adjoint_computed = True
            else:
                self.is_adjoint_computed = False
        elif type == "accept":
            if self.is_state_computed:
                self.ucache = self.uwork
            if self.is_adjoint_computed:
                self.pcache = self.pwork
        elif type == "temp":
            self.is_state_computed = False
            self.is_adjoint_computed = False
    
    def value(self, z, ftol):
        ferr = 0
        if not self.is_state_computed or self.uwork is None:
            self.uwork, cnt0, serr = self.con0.solve(z, ftol)
            self.cnt["ninvJ1"] += cnt0
            self.is_state_computed = True
            self.cnt["nstate"] += 1
            ferr = max(ferr, serr)
        val, verr = self.obj0.value([self.uwork, z], ftol)
        return val, max(ferr, verr)
    
    def gradient(self, z, gtol):
        gerr = 0
        if not self.is_state_computed or self.uwork is None:
            self.uwork, cnt0, serr = self.con0.solve(z, gtol)
            self.cnt["ninvJ1"] += cnt0
            self.is_state_computed = True
            self.cnt["nstate"] += 1
            gerr = max(gerr, serr)
        if not self.is_adjoint_computed or self.pwork is None:
            rhs, aerr1 = self.obj0.gradient_1([self.uwork, z], gtol)
            rhs = -rhs
            
            self.pwork, aerr2 = self.con0.apply_inverse_adjoint_jacobian_1(rhs, [self.uwork, z][0], gtol)
            self.cnt["ninvJ1"] += 1
            self.is_adjoint_computed = True
            self.cnt["nadjoint"] += 1
            gerr = max(gerr, aerr1, aerr2)
        Bp, jerr = self.con0.apply_adjoint_jacobian_2(self.pwork, [self.uwork, z], gtol)
        grad, gerr1 = self.obj0.gradient_2([self.uwork, z], gtol)
        return grad + Bp, max(gerr, jerr, gerr1)
    
    def hessVec(self,v,z,htol):
        herr = 0
        #solve the state equation for u(z)
        if not self.is_state_computed or self.uwork is None:
            self.uwrok, cnt0, serr = self.con0.solve(z,htol)
            self.cnt["ninvJ1"] += cnt0
            self.is_state_computed = True
            self.cnt["nstate"] += 1
            herr = max(herr, serr)
            
            
        #solve the adjoint equation for p
        if not self.is_adjoint_computed or self.pwork is None:
            rhs, aerr1 = self.obj0.gradient_1([self.uwork,z],htol)
            rhs = -rhs
            self.pwork, aerr2 = self.con0.apply_inverse_adjoint_jacobian_1(rhs, [self.uwork,z][0],htol)
            self.cnt["ninvJ1"] += 1
            self.is_adjoint_computed = True
            self.cnt["nadjoint"] += 1
            herr = max(gerr, aerr1, aerr2)
            
            
        #compute the directional derivative of the state (\delta u)
        rhs_du = -self.con0.apply_jacobian_2(v,[self.uwork,z],htol)[0]
        du, herr1 = self.con0.apply_inverse_jacobian_1(rhs_du, [self.uwork,z][0],htol)
        
        #compute the directional dervative of the adjoint (\delta p)
        rhs_dp = -self.obj0.hessVec_11(du,[self.uwork,z],htol)[0]-self.con0.apply_adjoint_hessian_11(du, self.pwork, [self.uwork,z],htol)[0]
        dp, herr2 = self.con0.apply_inverse_jacobian_1(rhs_dp, [self.uwork,z][0],htol)
        
        #compute the Hessian-vector product
        hv_z,herr3 = self.obj0.hessVec_22(v,[self.uwork,z],htol)
        hv_cross, herr4 = self.con0.apply_adjoint_jacobian_2(dp,[self.uwork,z],htol)
        hv = hv_z+hv_cross
        return hv, max(herr,herr1,herr2,herr3,herr4)
            
            
    
    def profile(self):
        print("\nProfile Reduced Objective")
        print("  #state    #adjoint    #statesens    #adjointsens    #linearsolves")
        print("  {:6d}      {:6d}        {:6d}          {:6d}           {:6d}".format(
            self.cnt["nstate"], self.cnt["nadjoint"], self.cnt["nstatesens"],
            self.cnt["nadjointsens"], self.cnt["ninvJ1"]
        ))
        self.cnt["con"] = self.con0.profile()
        return self.cnt
