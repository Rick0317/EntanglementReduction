import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tensor_train as matrix_product_state
from tensorly import tt_to_tensor
#from tensorly.decomposition import matrix_product_state

def tensor_to_exact_mps(tensor):
    # Ensure the input is a TensorLy tensor
    tensor = tl.tensor(tensor)
    
    # Check if the tensor is 3-dimensional
    if tl.ndim(tensor) != 3:
        raise ValueError("Input tensor must be 3-dimensional")
    
    # Get the shape of the tensor
    shape = tl.shape(tensor)
    
    # Calculate the maximum possible bond dimension
    max_bond_dim = min(shape[0] * shape[1], shape[1] * shape[2], shape[0] * shape[2])
    
    # Perform MPS decomposition with increasing bond dimensions
    for bond_dim in range(1, max_bond_dim + 1):
        factors = matrix_product_state(tensor, rank=bond_dim)
        reconstructed = tl.tt_to_tensor(factors)
        mps_differ = tensor - reconstructed
        # Check if the reconstruction is exact (within a small tolerance)
        if np.sum(np.abs(mps_differ)) < 1e-6:
            return bond_dim
               # "bond_dimension": bond_dim,
         #       "shapes": [tl.shape(factor) for factor in factors],
         #       "bond_dimensions": [tl.shape(factor)[0] for factor in factors[1:]] + [tl.shape(factors[-1])[-1]]
            
    
    # If we reach here, we couldn't find an exact representation
    raise ValueError("Could not find an exact MPS representation")

    
    
def tensor_to_exact_cp(tensor, max_rank=None, tol=1e-4, n_iter_max=100):
    # Ensure the input is a TensorLy tensor
    tensor = tl.tensor(tensor)
    
    # Check if the tensor is 3-dimensional
    if tl.ndim(tensor) != 3:
        raise ValueError("Input tensor must be 3-dimensional")
    
    # Get the shape of the tensor
    shape = tl.shape(tensor)
    
    # Calculate the maximum possible rank if not provided
    if max_rank is None:
        max_rank = min(shape[0] * shape[1], shape[0] * shape[2], shape[1] * shape[2])
    
    # Perform CP decomposition with increasing ranks
    for rank in range(1, max_rank + 1):
        factors = parafac(tensor, rank=rank, n_iter_max=n_iter_max, tol=tol, init="svd")
        reconstructed = tl.cp_to_tensor(factors)
        
        # Calculate the relative error
        error = tl.norm(tensor - reconstructed) / tl.norm(tensor)
        
        # Check if the reconstruction is exact (within the specified tolerance)
        if error < tol:
            return rank
    
    # If we reach here, we couldn't find an exact representation
    raise ValueError(f"Could not find an exact CP representation within rank {max_rank}")

## CP rank 3
def cp3(f1):
    cp_factors = parafac(f1, rank=3)
    
    tensor_output = tl.cp_to_tensor(cp_factors)

    # Reshape to n^3 by 1
    n = tensor_output.shape[0]  # assuming `tensor_output` is n x n x n
    reshaped_output = np.reshape(tensor_output, (n**3, 1))

    return reshaped_output

## CP rank 2
def cp2(f1):
    cp_factors = parafac(f1, rank=2)
    
    tensor_output = tl.cp_to_tensor(cp_factors)

    # Reshape to n^3 by 1
    n = tensor_output.shape[0]  # assuming `tensor_output` is n x n x n
    reshaped_output = np.reshape(tensor_output, (n**3, 1))

    return reshaped_output

## CP rank 1
def cp1(f1):
    cp_factors = parafac(f1, rank=1)
    
    tensor_output = tl.cp_to_tensor(cp_factors)

    # Reshape to n^3 by 1
    n = tensor_output.shape[0]  # assuming `tensor_output` is n x n x n
    reshaped_output = np.reshape(tensor_output, (n**3, 1))

    return reshaped_output

## MPS rank 5
def mp5(f1):
    mps_factors = matrix_product_state(f1, rank=[1,5,5,1])
    
    tensor_output = tt_to_tensor(mps_factors)

    # Reshape to n^3 by 1
    n = tensor_output.shape[0]  # assuming `tensor_output` is n x n x n
    reshaped_output = np.reshape(tensor_output, (n**3, 1))

    return reshaped_output



## MPS rank 3
def mp3(f1):
    mps_factors = matrix_product_state(f1, rank=[1,3,3,1])
    
    tensor_output = tt_to_tensor(mps_factors)

    # Reshape to n^3 by 1
    n = tensor_output.shape[0]  # assuming `tensor_output` is n x n x n
    reshaped_output = np.reshape(tensor_output, (n**3, 1))

    return reshaped_output

## MPS rank 2
def mp2(f1):
    mps_factors = matrix_product_state(f1, rank=[1,2,2,1])
    
    tensor_output = tt_to_tensor(mps_factors)

    # Reshape to n^3 by 1
    n = tensor_output.shape[0]  # assuming `tensor_output` is n x n x n
    reshaped_output = np.reshape(tensor_output, (n**3, 1))

    return reshaped_output

## MPS rank 1
def mp1(f1):
    mps_factors = matrix_product_state(f1, rank=[1,1,1,1])
    
    tensor_output = tt_to_tensor(mps_factors)

    # Reshape to n^3 by 1
    n = tensor_output.shape[0]  # assuming `tensor_output` is n x n x n
    reshaped_output = np.reshape(tensor_output, (n**3, 1))

    return reshaped_output

## MPS error rank 1
def mps1(f1):
    mps_factors = matrix_product_state(f1, rank=[1,1,1,1])
    mps_differ = f1-tt_to_tensor(mps_factors)
    
    return np.round(np.sum(np.abs(mps_differ)),7)


## MPS error rank 2
def mps2(f1):
    mps_factors = matrix_product_state(f1, rank=[1,2,2,1])
    mps_differ = f1-tt_to_tensor(mps_factors)

    return np.round(np.sum(np.abs(mps_differ)),7)


## CP error rank 1
def cpd1(f1):
    cp_factors = parafac(f1, rank=1)
    cp_differ= f1-tl.cp_to_tensor(cp_factors)
    
    return np.round(np.sum(np.abs(cp_differ)),7)

## CP error rank 2
def cpd2(f1):
    cp_factors = parafac(f1, rank=2)
    cp_differ= f1-tl.cp_to_tensor(cp_factors)
    
    return np.round(np.sum(np.abs(cp_differ)),7)


'''

def tensor_to_exact_mps(tensor):
    # Ensure the input is a TensorLy tensor
    tensor = tl.tensor(tensor)
    
    # Check if the tensor is 3-dimensional
    if tl.ndim(tensor) != 3:
        raise ValueError("Input tensor must be 3-dimensional")
    
    # Get the shape of the tensor
    shape = tl.shape(tensor)
    
    # Calculate the maximum possible bond dimension
    max_bond_dim = min(shape[0] * shape[1], shape[1] * shape[2], shape[0] * shape[2])
    
    # Perform MPS decomposition with increasing bond dimensions
    for bond_dim in range(1, max_bond_dim + 1):
        factors = matrix_product_state(tensor, rank=bond_dim)
        reconstructed = tl.tt_to_tensor(factors)
        
        # Check if the reconstruction is exact (within a small tolerance)
        if tl.norm(tensor - reconstructed) / tl.norm(tensor) < 1e-4:
            return {
                "factors": factors,
                "shapes": [tl.shape(factor) for factor in factors],
                "bond_dimensions": [tl.shape(factor)[0] for factor in factors[1:]] + [tl.shape(factors[-1])[-1]]
            }
    
    # If we reach here, we couldn't find an exact representation
    raise ValueError("Could not find an exact MPS representation")
'''