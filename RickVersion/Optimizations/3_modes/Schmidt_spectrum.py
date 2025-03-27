"""
This is the file for optimizing the Bogolibov unitary based on the
Schmidt spectrum of the new basis ground state.

Change the Model Hamiltonian that you use in get_Hamiltonian() function and
model = , variable definition
"""
from RickVersion.utils.util_gfro import *
from RickVersion.utils.util_tensornetwork import get_approx_bd_mps, get_mps
from RickVersion.utils.util_hamil import bilinear_three_mode_H, anharmonic_three_mode_H
from RickVersion.utils.util_mutualinfo import mutual_info_full
from openfermion import get_quad_operator
from scipy.optimize import minimize
from RickVersion.Graph_plots.mutual_information_visualizer import visualize_mutual_information
import csv
import os
import matplotlib.pyplot as plt

def get_Hamiltonian():
    h_variables = [1 / 2, 1 / 2, 1 / 2, 0.1, 0.1, 0.1]
    model_H = bilinear_three_mode_H(h_variables)
    return model_H


def shmidt_spectrum_cost(X, H, n, trunc, bd):

    P, Q = recreate_matrices_PQ_only(X, n)

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA, n)

    B_b, B_bd = transformed_op_inv_no_translation(U, V)

    model_H = get_Hamiltonian()

    H2 = substitute_operators(model_H, B_b, B_bd)

    # H1 = quad_diagonalization(H2, n)

    _, eig_vec = boson_eigenspectrum_sparse(H2, trunc, 1)

    ground_state = eig_vec

    matrix_1 = ground_state.reshape(10, 100)
    _, S1, __ = np.linalg.svd(matrix_1, full_matrices=False)
    sum_S1_first = np.sum(S1[:5])
    sum_s1_second = np.sum(S1[5:])
    inverse1 = 1 / (sum_S1_first - sum_s1_second)

    matrix_2 = ground_state.reshape(100, 10)
    _, S2, __ = np.linalg.svd(matrix_2, full_matrices=False)
    sum_S2_first = np.sum(S2[:5])
    sum_S2_second = np.sum(S2[5:])
    inverse2 = 1 / (sum_S2_first - sum_S2_second)

    return inverse1 + inverse2

def hamiltonian_reconstruction(X, H, n):
    P, Q = recreate_matrices_PQ_only(X, n)

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA, n)

    B_b, B_bd = transformed_op_inv_no_translation(U, V)

    H1 = substitute_operators(H, B_b, B_bd)

    return H1


if __name__ == '__main__':
    model = "bilinear_three_mode_H"
    truncation = 10
    model_H = get_Hamiltonian()

    eig_value, eig_vec = boson_eigenspectrum_sparse(model_H, truncation, 1)

    ed_ground_state = eig_vec

    ed_ground_state_energy = eig_value

    reshaped_gs = ed_ground_state.reshape(truncation, truncation, truncation)

    I_12, I_23, I_13 = mutual_info_full(reshaped_gs, truncation)
    visualize_mutual_information(I_12, I_23, I_13)

    filter_threshold = I_12 + I_23 + I_13

    mps1_error, _ = get_mps(reshaped_gs, 1)

    error_threshold_frac = 1e-4
    threshold = abs(ed_ground_state_energy * error_threshold_frac)
    exact_bd = get_approx_bd_mps(reshaped_gs, threshold=threshold)

    print(f"Almost Exact BD {threshold}: {exact_bd}")
    print(f"Ground state energy: {ed_ground_state_energy}")

    n = 3
    P, Q = initial_only_PQ(n)
    X = 1e-6 * flatten_matrices_only_PQ(P, Q, n)
    maxit = 200
    options = {
        'maxiter': maxit,
        'tol': 1e-30,
        'disp': False
    }

    bd = 1
    intermediate_values = []

    def cost_fn(X):
        trunc = 10
        cost1 = shmidt_spectrum_cost(X, model_H, n, trunc, bd)
        return cost1

    intermediate_data = []

    def printx(xk):
        current_value = cost_fn(xk)
        intermediate_data.append(current_value)
        intermediate_values.append(current_value)
        print("Current Schmidt spectrum sum:", current_value)


    # Minimize the cost function
    result = minimize(cost_fn, X, method='COBYLA', options=options, callback=printx)
    # print("Gradient norm:", np.linalg.norm(result.jac))
    truncation = 10
    model_H = get_Hamiltonian()

    H_optimized = hamiltonian_reconstruction(result.x, model_H, n)

    eig_value, eig_vecs = boson_eigenspectrum_sparse(H_optimized, truncation, 1)

    ground_state_optim = eig_vecs

    ground_state_energy_optim = eig_value

    reshaped_gs = ground_state_optim.reshape(truncation, truncation, truncation)

    I_12, I_23, I_13 = mutual_info_full(reshaped_gs, truncation)
    visualize_mutual_information(I_12, I_23, I_13)


    error, _ = get_mps(reshaped_gs, bd)

    energy_change = abs(ground_state_energy_optim - ed_ground_state_energy)
    print(f"Error in MPS: {error}")

    bd_optim = get_approx_bd_mps(reshaped_gs, threshold=threshold)
    print(f"Almost Exact BD {threshold}: {bd_optim}")
    print(f"Ground state energy: {ground_state_energy_optim}")
    print(f"Ground state energy change: {energy_change}")

    # Create an index list (0, 1, 2, ...)
    filtered_values = [val if val < filter_threshold else np.nan for val in
                       intermediate_values]

    # Create an index list
    indices = range(len(filtered_values))

    # Plot
    plt.plot(indices, filtered_values, marker='o', linestyle='-')

    # Labels and title
    plt.xlabel("Index")
    plt.ylabel("Intermediate Values (Filtered)")
    plt.title("Plot with Outliers Removed")

    # Show the plot
    plt.show()

    file_name = f"../../Results/Schmidt_spectrum.csv"

    file_exists = os.path.isfile(file_name)

    if energy_change < threshold:

        with open(file_name, mode='a' if file_exists else 'w', newline='',
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header only if the file doesn't exist
            if not file_exists:
                writer.writerow(
                    ['Model', 'Truncation', 'Threshold', 'Initial BD', 'Final BD', 'Method', 'Intermediate Data'])

            # Write the data
            writer.writerow(
                [model, truncation, threshold, exact_bd, bd_optim, 'Mutual Info', intermediate_data])
