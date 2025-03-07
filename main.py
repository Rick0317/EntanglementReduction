import time
import csv
import inspect
import numpy as np
from utils.util_hamil import test,test1,test2,test5
from utils.util_mutualinfo import mutual_all
from utils.util_covar import covariance
from utils.util_save import printx,save_parameters
from utils.util_gfro import (obtain_fragment,
                       rotated_hamiltonian,
                       boson_eigenspectrum_sparse,
                       get_boson_eigenspectrum_full,
                       quad_diagonalization)

def extract_function_name(func):
    return func.__name__


"""
Parameters for the experiments
n: number of modes
h_variables: list of coefficients in the Bosonic Hamiltonian for testing
truncation: The number of basis functions at each site
hamil_name: Name of the Hamiltonian used for testing
H: The Bosonic Hamiltonian being constructed
maxit: The maximum number of iterations in the optimization process
options: The options for the optimization process
cost_fns: The list of cost functions
"""

n = 3
h_variables = [1, 1, 1, 0.1, 0.1, 0.1]
truncation = 3
hamil_name = extract_function_name(test5)
H = test5(h_variables)
maxit = 10
options = {
    'maxiter': maxit,
    'gtol': 1e-7,
    'disp': False
}
start_time = time.time()

cost_fns = [
    # lambda i_12, i_23, i_13: i_12 + 1 * i_13 + i_23,
    # lambda i_12, i_23, i_13: i_12 + 2 * i_13 + i_23,
    # lambda i_12, i_23, i_13: i_12 + 3 * i_13 + i_23,
    # lambda i_12, i_23, i_13: i_12 + 4 * i_13 + i_23,
    # lambda i_12, i_23, i_13: i_12 + 5 * i_13 + i_23,
    # lambda i_12, i_23, i_13: i_12 + 6 * i_13 + i_23,
    # lambda i_12, i_23, i_13: i_12 + 7 * i_13 + i_23,
    # lambda i_12, i_23, i_13: i_12 + 8 * i_13 + i_23,
    # lambda i_12, i_23, i_13: i_12 + 9 * i_13 + i_23,
    lambda i_12, i_23, i_13: i_13,
    # lambda i_12, i_23, i_13: i_12 + 20 * i_13 + i_23,
]


"""
Pre optimization process
"""
eigenvalues1, e1 = boson_eigenspectrum_full(H, truncation)
e11 = e1[:, 0]
f1 = e11.reshape(truncation, truncation, truncation)
print(f"Ground state energy before rotation: {eigenvalues1[0]}")

I_12, I_23, I_13 = mutual_all(f1,truncation)
covar_XY, covar_YZ, covar_XZ = covariance(e11, truncation)

for i in range(truncation ** n - 1):
    ei = e1[:, i]
    fi = ei.reshape(truncation, truncation, truncation)

    I_12, I_23, I_13 = mutual_all(fi, truncation)
    covar_XY, covar_YZ, covar_XZ = covariance(ei, truncation)


parameters = []
parameters.append([round(I_12, 7), round(I_23, 7), round(I_13, 7), mps1(f1),
                   mps2(f1), cpd1(f1), cpd2(f1),
                   round(covar_XY, 7), round(covar_YZ, 7), round(covar_XZ, 7)])

print(f"Covariances before rotations: {covar_XY, covar_YZ, covar_XZ}")
print(f"MPS error for BD=truncation^(n // 2): {mpsN(f1, truncation ** (n // 2))}")

params = [round(I_12, 7), round(I_23, 7), round(I_13, 7),
          mps1(f1), mps2(f1), cpd1(f1), cpd2(f1),
          round(covar_XY, 7), round(covar_YZ, 7), round(covar_XZ, 7)]

iteration = 0

printx(params, iteration)


for fn in cost_fns:

    H = test5(h_variables)
    result = obtain_fragment(H, n, options, fn)

    Hr1 = rotated_hamiltonian(H, result.x.copy(), n)

    Hr2 = quad_diagonalization(Hr1, n)

    eigenvalues2, e2 = boson_eigenspectrum_full(Hr2, truncation)

    e2 = e2.copy()[:, 0]

    print(f"Rotated Eigenvalue: {eigenvalues2[0]}")

    f2 = e2.reshape(truncation, truncation, truncation)

    I_12, I_23, I_13 = mutual_all(f2.copy(), truncation)
    covar_XY, covar_YZ, covar_XZ = covariance(e2.copy(), truncation)

    params = [round(I_12, 7), round(I_23, 7), round(I_13, 7),
              mps1(f2), mps2(f2), cpd1(f2), cpd2(f2),
              round(covar_XY, 7), round(covar_YZ, 7), round(covar_XZ, 7)]

    parameters.append(params)

    printx(params, iteration)

    print(f"Covariances after rotation: {covar_XY, covar_YZ, covar_XZ}")

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time)
print("Completed")


save_parameters(cost_fns, h_variables, maxit, parameters, hamil_name)


