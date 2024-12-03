#In this file, we follow Section 3 to give the level-dependent objective function
#construction, where the smooth part keep the construction but in different 
#dimension, and the nonsmooth part is according to Definition 3.2
import numpy as np
def obj_nonsmooth(phi,R):
    """
    We construct phi_{i-1} according to Def 3.2
    phi_{i-1}(Rx) := phi_i(x)

    Parameters
    ----------
    phi : function
        phi_i.
    R : matrix
        restriction matrix from transfer_operator.py.

    Returns
    -------
    phi_{i-1}

    """
    def g(v):
        x = np.dot(R.T,v)
        return phi(x)
    return g



#The following test is same as Example 3.3
#def phi_try(x):
#    return np.linalg.norm(x, ord=1)
#def f_try(x):
#    return np.linalg.norm(x, ord=2)**2


       
#x_try = np.array([1, 2, 3, 0, 5, 0])
#def restriction_R(m,n):
#    matrix_R = np.zeros((m,n))
#    for i in range(m):
#        matrix_R[i,2*(i+1)-1] = 1/np.sqrt(2)
#        matrix_R[i,2*i] = 1/np.sqrt(2)
#    return matrix_R

#R_try = restriction_R(3, 6)
#print(phi_try(x_try))
#print(R_try@x_try)
#print(obj_nonsmooth(phi_try, R_try)(R_try@x_try))
    
#print(f_try(x_try))
#print(f_try(R_try@x_try))
