from RickVersion.utils.util_gfro import *
from RickVersion.utils.util_tensornetwork import get_approx_bd_mps, get_mps
from RickVersion.utils.util_hamil import bilinear_three_mode_H, anharmonic_three_mode_H, henon_heiles_two_mode_H
from RickVersion.utils.util_mutualinfo import mutual_info_full, mutual_info_full_n_sites
from openfermion import get_boson_operator
from scipy.optimize import minimize
from RickVersion.Graph_plots.mutual_information_visualizer import visualize_mutual_information
import matplotlib.pyplot as plt
from RickVersion.utils.ham_utils import get_H_op, get_H_ground_state, get_Hmol_params, get_H_ground_state_bosonic, truncate_quad_operator
from RickVersion.utils.opt_utils import optimize_bog_transfrom, modified_H_params, weighted_sum_MI


def get_hamiltonian(H_params):
    # Get H as open fermion quad-operator
    H = get_H_op(H_params)

    return get_boson_operator(H)


def mutual_info_cost_func(X, H, n, trunc, bd, n_modes, H_params):
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

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA,n)

    B_b, B_bd = transformed_op_inv_no_translation(U, V)

    model_H = get_hamiltonian(H_params)

    H1 = substitute_operators(model_H, B_b, B_bd)

    # H2 = quad_diagonalization(H1, n)
    del model_H, B_b, B_bd, U, V, expA, P, Q

    _, eig_vecs = boson_eigenspectrum_sparse(H1, trunc, 1)

    ground_state = eig_vecs

    reshaped_gs = ground_state.reshape(*(trunc,) * n_modes)

    MI_list = mutual_info_full_n_sites(reshaped_gs, trunc, n_modes)
    del reshaped_gs

    return weighted_sum_MI(MI_list, n)


def hamiltonian_reconstruction(X, H, n):
    P, Q = recreate_matrices_PQ_only(X, n)

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA, n)

    B_b, B_bd = transformed_op_inv_no_translation(U, V)

    H1 = substitute_operators(H, B_b, B_bd)

    return H1


if __name__ == '__main__':
    au_to_cm = 219474.63068
    mol = 'H2S'
    # Read in molecular Hamiltonian
    n_modes, H_params = get_Hmol_params(mol)
    truncation = 7
    error_threshold_frac = 1e-4

    model_H = get_hamiltonian(H_params)

    eig_value, eig_vec = boson_eigenspectrum_sparse(model_H, truncation, 1)

    ed_ground_state = eig_vec

    ed_ground_state_energy = eig_value

    reshaped_gs = ed_ground_state.reshape(*(truncation,) * n_modes)

    I_list = mutual_info_full_n_sites(reshaped_gs, truncation, n_modes)
    visualize_mutual_information(I_list, n_modes)

    filter_threshold = sum(I_list)

    mps1_error, _ = get_mps(reshaped_gs, 1)

    error_threshold_frac = 1e-4
    threshold = abs(ed_ground_state_energy * error_threshold_frac)
    exact_bd = get_approx_bd_mps(reshaped_gs, threshold=threshold)

    print(f"Almost Exact BD {threshold}: {exact_bd}")
    print(f"Ground state energy: {ed_ground_state_energy * au_to_cm}")

    P, Q = initial_only_PQ(n_modes)
    X = 1e-6 * flatten_matrices_only_PQ(P, Q, n_modes)
    maxit = 100
    options = {
        'maxiter': maxit,
        'tol': 1e-100,
        'disp': False
    }

    bd = 1
    intermediate_values = []

    def cost_fn(X):
        trunc = 7
        cost1 = mutual_info_cost_func(X, model_H, n_modes, trunc, bd, n_modes, H_params)
        return cost1

    intermediate_data = []

    def printx(xk):
        current_value = cost_fn(xk)
        intermediate_data.append(current_value)
        intermediate_values.append(current_value)
        print("Current total mutual information:", current_value)


    # Minimize the cost function
    result = minimize(cost_fn, X, method='COBYLA', options=options, callback=printx)

    model_H = get_hamiltonian(H_params)
    truncation = 7

    H_optimized = hamiltonian_reconstruction(result.x, model_H, n_modes)

    eig_value, eig_vecs = boson_eigenspectrum_sparse(H_optimized, truncation, 1)

    ground_state_optim = eig_vecs

    ground_state_energy_optim = eig_value

    reshaped_gs = ground_state_optim.reshape(*(truncation,) * n_modes)

    MI_list = mutual_info_full_n_sites(reshaped_gs, truncation, n_modes)

    visualize_mutual_information(MI_list, n_modes)

    error, _ = get_mps(reshaped_gs, bd)

    energy_change = abs(ground_state_energy_optim - ed_ground_state_energy)
    print(f"Error in MPS: {error}")

    bd_optim = get_approx_bd_mps(reshaped_gs, threshold=threshold)
    print(f"Almost Exact BD {threshold}: {bd_optim}")
    print(f"Ground state energy: {ground_state_energy_optim * au_to_cm}")
    print(f"Ground state energy change: {energy_change * au_to_cm}")

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
