import numpy as np
import time
from util_hamil import test,test1,test2,test4,test5,test7,test6
from util_mutualinfo import mutual_all
from util_covar import covariance,covariance_x2
from ten_network import mp2,cp2,mps2,cpd2,tensor_to_exact_mps,mps1,cpd1#,mp2,mp1,mp3,cp1,cp2,cp3#,tensor_to_mps


from openfermion.ops import BosonOperator
from openfermion.transforms import normal_ordered
from openfermion import get_sparse_operator

from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy import linalg
from scipy.linalg import eigh
from scipy.linalg import eig

import matplotlib.pyplot as plt

import csv

import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.decomposition import tensor_train 
from tensorly import tt_to_tensor
#from tensorly.decomposition import matrix_product_state

from util_bogo import (rotated_hamiltonian,quad_diagonalization,boson_eigenspectrum_full,initial,
                      flatten_matrices,recreate_matrices,super_matrix_exp,extract_sub_matrices,
                      tranformed_operators_inv,substitute_operators,initial_no_s_d)


def cost(X):
    P, Q, G, L = recreate_matrices(X, n)  # Calculate P,Q,G,L from paramter vector X
    expA = super_matrix_exp(P, Q)     # Calculate exp([[P,Q],[Q,P]])
    U, V = extract_sub_matrices(expA, n)     # Extract U and V from exp([[P,Q],[Q,P]]) = [[U.T,-V.T],[-V.T,U.T]]
    U = U.T
    V = -V.T            
    
    Q = np.zeros((n, n))  # No squeezing
    G = np.zeros(n)       # No displacement
 
    B_b, B_bd = tranformed_operators_inv(U, V, G)  # Calculate transformed operators 
    H1 = substitute_operators(H, B_b, B_bd)  # Hamitonian after rotation in to transformed operators
    H1 = quad_diagonalization(H1, n)   # Diagonalize up to quadratic part of the transformed Hamiltonian

    e1, ev = boson_eigenspectrum_full(H1, truncation)
    ev = ev[:,0]
    f1 = ev.reshape(truncation, truncation, truncation)


    fx = mp2(f1).reshape(truncation, truncation, truncation)
    I_12, I_23, I_13 = mutual_all(fx, truncation)
    reg = 1e-4 * np.linalg.norm(X)**2    # regularizing parameter
    cost = I_12 + I_23 + 5 * I_13 + reg

    return cost

      

#  Initialization 

n = 3 # number of modes
truncation = 6

# Define Hamiltonian using parameters
h_variables = [1,1,1,0.2,0.2,0.2,0.6,0.6,0.6,0.6,0.6,0.6]

H = test5(h_variables)

# Get ground state MPS bond dimension before rotation
eigenvalues1, e1 = boson_eigenspectrum_full(H, truncation)
e1 = e1[:,0]
f1 = e1.reshape(truncation, truncation, truncation)



# Initial guess
P, Q, G, L =  initial(n) 
#P, Q, G, L = initial_no_s_d(n) #for no squeezing and displacement

X = 1e-3 * flatten_matrices(P, Q, G, L, n)

options = {'maxiter': 100, 'disp': True}

# Run optimization
result = minimize(cost, X, method='BFGS', tol=1e-6, options=options)

# Post-Optimization Analysis 

'''
# for no squeezing and displacement

P, Q, G, L =  recreate_matrices(result.x, n)

Q = np.zeros((n, n))  # No squeezing
G = np.zeros(n)       # No displacement

X =  flatten_matrices(P, Q, G, L, n)

Hr = rotated_hamiltonian(H, X, n)
'''

Hr = rotated_hamiltonian(H, result.x, n)
Hr = quad_diagonalization(Hr, n)

# Get ground state MPS bond dimension after rotation
eigenvalues2, e2 = boson_eigenspectrum_full(Hr, truncation)
e2 = e2[:,0]
f2 = e2.reshape(truncation, truncation, truncation)

# Results
print("MPS BD representation error before rotation: ",mps2(f1))
print("MPS BD representation error after rotation: ",mps2(f2))





