def restriction_R(m,n):
    """
    Construct a sparse orthonormal matrix R in R^{m\times n} 

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns (n>>m).

    Returns
    -------
    matrix_R : TYPE
        DESCRIPTION.

    """
    matrix_R = np.zeros((m,n))
    for i in range(m):
        matrix_R[i,2*(i+1)-1] = 1/np.sqrt(2)
        matrix_R[i,2*i] = 1/np.sqrt(2)
    return matrix_R
    
