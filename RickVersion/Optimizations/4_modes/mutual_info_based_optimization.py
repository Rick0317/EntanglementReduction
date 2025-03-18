from RickVersion.utils.util_tensornetwork import get_approx_bd_mps, get_mps
from RickVersion.utils.util_hamil import four_mode_anharmonic_H
from RickVersion.utils.util_mutualinfo_4mode import mutual_info_full


def mutual_info_cost_func(X, H, n, trunc, bd):
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

    h_variables = [1 / 2, 1 / 2, 1 / 2, 1 / 2, 0.4]
    model_H = four_mode_anharmonic_H(h_variables)

    H1 = substitute_operators(model_H, B_b, B_bd)

    _, eig_vecs = boson_eigenspectrum_sparse(H1, trunc, 1)

    ground_state = eig_vecs

    reshaped_gs = ground_state.reshape(trunc, trunc, trunc, trunc)

    I_12, I_13, I_14, I_23, I_24, I_34 = mutual_info_full(reshaped_gs, n)

    return I_12 + I_13 + I_14 + I_23 + I_24 + I_34

def hamiltonian_reconstruction(X, H, n):
    P, Q = recreate_matrices_PQ_only(X, n)

    expA = super_matrix_exp(P, Q)

    U, V = extract_sub_matrices(expA, n)

    B_b, B_bd = transformed_op_inv_no_translation(U, V)

    H1 = substitute_operators(H, B_b, B_bd)

    return H1


if __name__ == '__main__':

    truncation = 8
    h_variables = [1 / 2, 1 / 2, 1 / 2, 1 / 2, 0.4]
    model_H = four_mode_anharmonic_H(h_variables)

    eig_value, eig_vec = boson_eigenspectrum_sparse(model_H, truncation, 1)

    ed_ground_state = eig_vec

    ed_ground_state_energy = eig_value

    reshaped_gs = ed_ground_state.reshape(truncation, truncation, truncation, truncation)
    threshold = 1e-10
    exact_bd = get_approx_bd_mps(reshaped_gs, threshold=threshold)

    print(f"Almost Exact BD {threshold}: {exact_bd}")
    print(f"Ground state energy: {ed_ground_state_energy}")

    n = 4
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
        cost1 = mutual_info_cost_func(X, model_H, n, truncation, bd)
        return cost1

    def printx(xk):
        current_value = cost_fn(xk)
        print("Current total mutual information:", current_value)


    # Minimize the cost function
    result = minimize(cost_fn, X, method='BFGS', tol=None,
                      options=options, callback=printx)

    h_variables = [1 / 2, 1 / 2, 1 / 2, 1 / 2, 0.4]
    model_H = four_mode_anharmonic_H(h_variables)

    H_optimized = hamiltonian_reconstruction(result.x, model_H, n)

    eig_value, eig_vecs = boson_eigenspectrum_sparse(H_optimized, truncation, 1)

    ground_state_optim = eig_vecs

    ground_state_energy_optim = eig_value

    reshaped_gs = ground_state_optim.reshape(truncation, truncation, truncation, truncation)

    bd_optim = get_approx_bd_mps(reshaped_gs, threshold=threshold)

    error, _ = get_mps(reshaped_gs, bd)

    print(f"Error in MPS: {error}")

    print(f"Almost Exact BD {threshold}: {bd_optim}")
    print(f"Ground state energy: {ground_state_energy_optim}")
    print(f"Ground state energy change: {abs(ground_state_energy_optim - ed_ground_state_energy)}")
