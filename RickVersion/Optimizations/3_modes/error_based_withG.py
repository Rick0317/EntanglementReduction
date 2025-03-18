"""
This is the file for optimizing the Bogolibov unitary based on the error
of representing the ground state with finite bond dimension MPS.

Change the Model Hamiltonian that you use in get_Hamiltonian() function and
model = , variable definition
"""

from RickVersion.utils_temporary.util_gfro import *
from RickVersion.utils.util_tensornetwork import get_approx_bd_mps, get_mps
from RickVersion.utils.util_hamil import anharmonic_three_mode_H
from scipy.optimize import minimize
import csv
import os


def get_Hamiltonian():
    h_variables = [1 / 2, 1 / 2, 1 / 2, 0.4, 0.4, 0.4]
    model_H = anharmonic_three_mode_H(h_variables)
    return model_H


def error_based_costfn(X, H, n, trunc, bd):
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
    P,Q,G,L = recreate_matrices(X, n)

    expA = super_matrix_exp(P, Q)  # Calculate exp([[P,Q],[Q,P]])

    U, V = extract_sub_matrices(expA,n)

    B_b, B_bd = tranformed_operators_inv(U, V, G)

    model_H = get_Hamiltonian()

    H1 = substitute_operators(model_H, B_b, B_bd)

    H2 = quad_diagonalization(H1, n)

    _, eig_vecs = boson_eigenspectrum_sparse(H2, trunc, 1)

    ground_state = eig_vecs

    # 3 modes case
    reshaped_gs = ground_state.reshape(trunc, trunc, trunc)

    error, _ = get_mps(reshaped_gs, bd)

    return error


def hamiltonian_reconstruction(X, H, n):
    """
    Construct the optimized Hamiltonian with the new parameters X
    :param X:
    :param H:
    :param n:
    :return:
    """
    P,Q,G,L = recreate_matrices(X, n)

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA, n)

    relation1 = U.conj().T @ U - V.conj().T @ V
    relation2 = U.conj().T @ V - V.conj().T @ U

    assert np.allclose(relation1, np.eye(n)), "Relation1 is incorrect"
    assert np.allclose(relation2, np.zeros((n, n))), "Relation2 is incorrect"

    B_b, B_bd = tranformed_operators_inv(U, V, G)

    H1 = substitute_operators(H, B_b, B_bd)

    return H1


def rotation_unitarity_check(X, n):
    """
    Checks if the parameters for U and V make sense.
    :param X:
    :param n:
    :return:
    """
    P,Q,G,L = recreate_matrices(X, n)

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA, n)

    relation1 = U.conj().T @ U - V.conj().T @ V
    relation2 = U.conj().T @ V - V.conj().T @ U

    isFirstRelation = np.allclose(relation1, np.eye(n))
    isSecondRelation = np.allclose(relation2, np.zeros((n, n)))

    return isFirstRelation, isSecondRelation


if __name__ == '__main__':
    model = "anharmonic_three_mode_H"

    truncation = 5
    model_H = get_Hamiltonian()

    eig_value, eig_vec = boson_eigenspectrum_sparse(model_H, truncation, 1)

    ed_ground_state = eig_vec

    ed_ground_state_energy = eig_value

    reshaped_gs = ed_ground_state.reshape(truncation, truncation, truncation)
    threshold = abs(ed_ground_state_energy) * 0.0001

    exact_bd = get_approx_bd_mps(reshaped_gs, threshold=threshold)

    print(f"Almost Exact BD {threshold}: {exact_bd}")
    print(f"Ground state energy: {ed_ground_state_energy}")

    n = 3
    P, Q, G, L = initial(n)
    X = 1e-6 * flatten_matrices(P, Q, G, L, n)
    maxit = 10
    options = {
        'maxiter': maxit,
        'gtol': 1e-7,
        'disp': False
    }

    bd = 2

    def cost_fn(X):
        cost1 = error_based_costfn(X, model_H, n, truncation, bd)
        return cost1


    intermediate_data = []

    def printx(xk):
        current_value = cost_fn(xk)
        intermediate_data.append(current_value)
        print("Current error:", current_value)

    result = minimize(cost_fn, X, method='BFGS', options=options, callback=printx)

    model_H = get_Hamiltonian()
    truncation = 10

    H_optimized = hamiltonian_reconstruction(result.x, model_H, n)

    check1, check2 = rotation_unitarity_check(result.x, n)
    print(f"Unitarity checks results: {check1} and {check2}")

    eig_value, eig_vecs = boson_eigenspectrum_sparse(H_optimized, truncation, 1)

    ground_state_optim = eig_vecs

    ground_state_energy_optim = eig_value

    reshaped_gs = ground_state_optim.reshape(truncation, truncation, truncation)

    bd_optim = get_approx_bd_mps(reshaped_gs, threshold=threshold)

    error, _ = get_mps(reshaped_gs, bd)

    change_gs = abs(ground_state_energy_optim - ed_ground_state_energy)

    print(f"Error in MPS: {error}")

    print(f"Almost Exact BD {threshold}: {bd_optim}")
    print(f"Ground state energy: {ground_state_energy_optim}")
    print(f"Ground state energy change: {change_gs}")


    file_name = f"../../../Results/error_based_3mod.csv"

    file_exists = os.path.isfile(file_name)

    if change_gs < abs(ed_ground_state_energy) * 0.0001:

        with open(file_name, mode='a' if file_exists else 'w', newline='',
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header only if the file doesn't exist
            if not file_exists:
                writer.writerow(
                    ['Model', 'Truncation', 'Threshold', 'Initial BD', 'Final BD',
                     'Method', 'Maximum Iterations', 'Intermediate Data'])

            # Write the data
            writer.writerow(
                [model, truncation, threshold, exact_bd, bd_optim, 'Representation Error',
                 maxit, intermediate_data])
