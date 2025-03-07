import numpy as np

def tensor_product(*matrices):
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result

def x1(n_max):
    m=1
    hbar=1
    omega=1
    """Create the position operator matrix for a single mode."""
    x = np.zeros((n_max, n_max))
    prefactor = np.sqrt(hbar / (2 * m * omega))
    for i in range(n_max - 1):
        x[i, i + 1] = x[i + 1, i] = prefactor * np.sqrt(i + 1)
    return x

def x2(n_max):
    m = 1
    hbar = 1
    omega = 1
    """Create the x^2 operator matrix for a single mode."""
    x_squared = np.zeros((n_max, n_max))
    prefactor = hbar / (2 * m * omega)
    
    for i in range(n_max - 1):
        # Diagonal elements (from squaring position operator x)
        x_squared[i, i] += prefactor * (2 * (i + 1))
        
        # Off-diagonal elements (from x^2 operator)
        if i < n_max - 2:
            x_squared[i, i + 2] = x_squared[i + 2, i] = prefactor * np.sqrt((i + 1) * (i + 2))

    return x_squared


def covariance(e1,truncation):

    X_mat = tensor_product(x1(truncation),np.eye(truncation))
    Y_mat = tensor_product(np.eye(truncation),x1(truncation))

    
    exp_val_X = np.einsum('i,ij,j', e1.conj(), X_mat, e1).real
    exp_val_Y = np.einsum('i,ij,j', e1.conj(), Y_mat, e1).real
 
    
    XY_mat = tensor_product(x1(truncation),x1(truncation))
 
    
    exp_val_XY = np.einsum('i,ij,j', e1.conj(), XY_mat, e1).real

    
    covar_XY = exp_val_XY - exp_val_X * exp_val_Y

    
    return covar_XY

def covariance_x2(e1,truncation):

    X_mat = tensor_product(x2(truncation),np.eye(truncation))
    Y_mat = tensor_product(np.eye(truncation),x2(truncation))
    
    exp_val_X = np.einsum('i,ij,j', e1.conj(), X_mat, e1).real
    exp_val_Y = np.einsum('i,ij,j', e1.conj(), Y_mat, e1).real

    
    XY_mat = tensor_product(x2(truncation),x2(truncation))

    
    exp_val_XY = np.einsum('i,ij,j', e1.conj(), XY_mat, e1).real

    
    covar_XY = exp_val_XY - exp_val_X * exp_val_Y

    
    return covar_XY



def covariance_x21(e1,truncation):

    X_mat = tensor_product(x2(truncation),np.eye(truncation))
    Y_mat = tensor_product(np.eye(truncation),x1(truncation))
    
    exp_val_X = np.einsum('i,ij,j', e1.conj(), X_mat, e1).real
    exp_val_Y = np.einsum('i,ij,j', e1.conj(), Y_mat, e1).real

    
    XY_mat = tensor_product(x2(truncation),x1(truncation))

    
    exp_val_XY = np.einsum('i,ij,j', e1.conj(), XY_mat, e1).real

    
    covar_XY = exp_val_XY - exp_val_X * exp_val_Y

    
    return covar_XY


def covar_cost_x2(e1,truncation,fn):
    
    I_12,I_23,I_13 = covariance_x2(e1,truncation)
    
    return fn(I_12, I_23, I_13)


def covar_cost(e1,truncation,fn):
    
    I_12,I_23,I_13 = covariance(e1,truncation)
    
    return fn(I_12, I_23, I_13) #abs(I_12)+5*abs(I_13)+abs(I_23)  #