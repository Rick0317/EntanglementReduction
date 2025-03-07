"""
This is the file for optimizing the Bogolibov unitary based on the
minimum energy of the finite bond dimension MPS representation of the
rotated ground state.

Change the Model Hamiltonian that you use in get_Hamiltonian() function and
model = , variable definition
"""

from utils.util_gfro import *
from utils.util_tensornetwork import get_approx_tensor, get_exact_bd_mps, get_approx_bd_mps, get_mps
from utils.util_hamil import bilinear_three_mode_H, anharmonic_three_mode_H
from openfermion import expectation, get_sparse_operator
import numpy as np
from scipy.optimize import minimize
import csv
import os


def get_Hamiltonian():
    h_variables = [1 / 2, 1 / 2, 1 / 2, 0.6]
    model_H = anharmonic_three_mode_H(h_variables)
    return model_H


def energy_based_costfn(X, H, n, trunc, bd):
    """
    We find the lowest energy with a given bond dimension
    We need to find the state with bd that lowers the energy.
    how to find such state.
    :param X:
    :param H:
    :param n:
    :param trunc:
    :param bd: The bond dimension we use to find the lowest energy
    :return:
    """
    P, Q = recreate_matrices_PQ_only(X, n)

    expA = super_matrix_exp(P, Q)  # Calculate exp([[P,Q],[Q,P]])

    U, V = extract_sub_matrices(expA,n)

    B_b, B_bd = transformed_op_inv_no_translation(U, V)

    model_H = get_Hamiltonian()

    H1 = substitute_operators(model_H, B_b, B_bd)

    _, eig_vecs = boson_eigenspectrum_sparse(H1, trunc, 1)

    ground_state = eig_vecs


    # 3 modes case
    reshaped_gs = ground_state.reshape(trunc, trunc, trunc)

    approximate_gs = get_approx_tensor(reshaped_gs, bd)

    reshaped_approximate = approximate_gs.reshape(trunc ** 3,)

    sparse_op = get_sparse_operator(H1, trunc=trunc)
    sparse_op_array = sparse_op.toarray()

    exp_value = np.vdot(reshaped_approximate, sparse_op_array @ reshaped_approximate)
    norm_factor = np.vdot(reshaped_approximate, reshaped_approximate)
    normalized_exp_value = exp_value / norm_factor

    return normalized_exp_value

def hamiltonian_reconstruction(X, H, n):
    P, Q = recreate_matrices_PQ_only(X, n)

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA, n)

    B_b, B_bd = transformed_op_inv_no_translation(U, V)

    H1 = substitute_operators(H, B_b, B_bd)

    return H1


if __name__ == '__main__':

    model = "anharmonic_three_mode_H"

    truncation = 10
    model_H = get_Hamiltonian()

    eig_value, eig_vec = boson_eigenspectrum_sparse(model_H, truncation, 1)

    ed_ground_state = eig_vec

    ed_ground_state_energy = eig_value

    reshaped_gs = ed_ground_state.reshape(truncation, truncation, truncation)

    threshold = abs(eig_value * 0.0001)

    exact_bd = get_approx_bd_mps(reshaped_gs, threshold=threshold)

    print(f"Almost Exact BD {threshold}: {exact_bd}")
    print(f"Ground state energy: {ed_ground_state_energy}")

    n = 3
    P, Q= initial_only_PQ(n)
    X = 1e-6 * flatten_matrices_only_PQ(P, Q, n)
    maxit = 10
    options = {
        'maxiter': maxit,
        'gtol': 1e-7,
        'disp': False
    }

    bd = 2

    def cost_fn(X):
        cost1 = energy_based_costfn(X, model_H, n, truncation, bd)
        return cost1


    intermediate_data = []
    def printx(xk):
        current_value = cost_fn(xk)
        intermediate_data.append(current_value)
        print("Current energy:", current_value)
        print(f"Ground state energy: {ed_ground_state_energy}")
        print(f"Difference: {abs(current_value - ed_ground_state_energy)}")

    result = minimize(cost_fn, X, method='BFGS', tol=None,
                      options=options, callback=printx)

    model_H = get_Hamiltonian()

    H_optimized = hamiltonian_reconstruction(result.x, model_H, n)

    eig_value, eig_vecs = boson_eigenspectrum_sparse(H_optimized, truncation, 1)

    ground_state_optim = eig_vecs

    ground_state_energy_optim = eig_value

    reshaped_gs = ground_state_optim.reshape(truncation, truncation, truncation)

    bd_optim = get_approx_bd_mps(reshaped_gs, threshold=threshold)

    error, _ = get_mps(reshaped_gs, bd)

    energy_change = abs(ground_state_energy_optim - ed_ground_state_energy)
    print(f"Error in MPS: {error}")

    print(f"Almost Exact BD {threshold}: {bd_optim}")
    print(f"Ground state energy: {ground_state_energy_optim}")
    print(f"Ground state energy change: {energy_change}")

    file_name = f"../../Results/energy_based_3mod.csv"

    file_exists = os.path.isfile(file_name)

    if energy_change < threshold:
        with open(file_name, mode='a' if file_exists else 'w', newline='',
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header only if the file doesn't exist
            if not file_exists:
                writer.writerow(
                    ['Model', 'Truncation', 'Threshold', 'Initial BD', 'Final BD',
                     'Method', 'Intermediate Data'])

            # Write the data
            writer.writerow(
                [model, truncation, threshold, exact_bd, bd_optim,
                 'Energy Error',
                 intermediate_data])

