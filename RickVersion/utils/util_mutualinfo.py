# These functions are for 3 modes systems.
import numpy as np


def von_neumann_entropy(rho):
    """
    Calculates the von Neumann entropy of a given density matrix.
    :param rho:
    :return:
    """
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))


def two_mode_rdm(density_matrix, i, j, trunc):
    """
    Find the 2 modes reduced density matrix of a given density matrix.
    :param density_matrix: Density matrix
    :param i: The first mode
    :param j: The second mode
    :param trunc: The truncation
    :return:
    """

    rho_ij = np.zeros((trunc ** 2, trunc ** 2), dtype=complex)

    # Determine which mode is being traced out
    trace_out_mode = 3 - (i + j)

    for a in range(trunc):
        for b in range(trunc):
            for c in range(trunc):
                for d in range(trunc):
                    # Using `np.sum` to trace out the `trace_out_mode` mode
                    if trace_out_mode == 0:
                        rho_ij[a * trunc + b, c * trunc + d] = np.sum(
                            density_matrix[:, a, b] * np.conj(density_matrix[:, c, d]))
                    elif trace_out_mode == 1:
                        rho_ij[a * trunc + b, c * trunc + d] = np.sum(
                            density_matrix[a, :, b] * np.conj(density_matrix[c, :, d]))
                    elif trace_out_mode == 2:
                        rho_ij[a * trunc + b, c * trunc + d] = np.sum(
                            density_matrix[a, b, :] * np.conj(density_matrix[c, d, :]))

    return rho_ij


def mutual_information(rho_i, rho_j, rho_ij):
    s_i = von_neumann_entropy(rho_i)
    s_j = von_neumann_entropy(rho_j)
    s_ij = von_neumann_entropy(rho_ij)
    return s_i + s_j - s_ij


# single-mode RDMs

def one_mode_rdm(density_matrix, trunc):
    """
    Find the 1 mode reduced density matrix of a given density matrix.
    :param density_matrix: The density matrix
    :param trunc:
    :return:
    """

    rho_1 = np.zeros((trunc, trunc), dtype=complex)
    rho_2 = np.zeros((trunc, trunc), dtype=complex)
    rho_3 = np.zeros((trunc, trunc), dtype=complex)

    for i in range(trunc):
        for j in range(trunc):
            rho_1[i, j] = np.sum(density_matrix[i, :, :] * np.conj(density_matrix[j, :, :]))
            rho_2[i, j] = np.sum(density_matrix[:, i, :] * np.conj(density_matrix[:, j, :]))
            rho_3[i, j] = np.sum(density_matrix[:, :, i] * np.conj(density_matrix[:, :, j]))

    return rho_1, rho_2, rho_3



def mutual_info_full(density_matrix, trunc):
    """
    Calculates the mutual information of a given density matrix for 3 modes
    :param density_matrix:
    :param trunc:
    :return:
    """

    rho_1, rho_2, rho_3 = one_mode_rdm(density_matrix, trunc)
    rho_12 = two_mode_rdm(density_matrix, 0, 1, trunc)
    rho_13 = two_mode_rdm(density_matrix, 0, 2, trunc)
    rho_23 = two_mode_rdm(density_matrix, 1, 2, trunc)

    I_12 = mutual_information(rho_1, rho_2, rho_12)
    I_23 = mutual_information(rho_2, rho_3, rho_23)
    I_13 = mutual_information(rho_1, rho_3, rho_13)

    return I_12, I_23, I_13

def single_entropy_full(f1, n):

    rho_1, rho_2, rho_3 = one_mode_rdm(f1, n)

    s1 = von_neumann_entropy(rho_1)
    s2 = von_neumann_entropy(rho_2)
    s3 = von_neumann_entropy(rho_3)

    return s1, s2, s3


def one_mode_rdm_n_sites(density_matrix, trunc, n):
    """
    Find the 1 mode reduced density matrix of a given density matrix.
    :param density_matrix: The density matrix
    :param trunc:
    :param n: Number of modes
    :return:
    """

    rho_list = [np.zeros((trunc, trunc), dtype=complex) for _ in range(n)]
    for site in range(n):
        for i in range(trunc):
            for j in range(trunc):
                idx = tuple(
                    i if ax == site else slice(None) for ax in range(n))
                idx_conj = tuple(
                    j if ax == site else slice(None) for ax in range(n))

                rho_list[site][i, j] = np.sum(density_matrix[idx] * np.conj(density_matrix[idx_conj]))

    return rho_list


def two_mode_rdm_n_sites(density_matrix, i, j, trunc, n):
    """
    Find the 2 modes reduced density matrix of a given density matrix.
    :param density_matrix: Density matrix
    :param i: The first mode
    :param j: The second mode
    :param trunc: The truncation
    :param n: Number of modes
    :return:
    """

    rho_ij = np.zeros((trunc ** 2, trunc ** 2), dtype=complex)

    for a in range(trunc):
        for b in range(trunc):
            for c in range(trunc):
                for d in range(trunc):
                    idx = tuple(
                        a if ax == i else b if ax == j else slice(None) for ax
                        in range(n)
                    )
                    idx_conj = tuple(
                        c if ax == i else d if ax == j else slice(None) for ax
                        in range(n)
                    )

                    rho_ij[a * trunc + b, c * trunc + d] = np.sum(
                        density_matrix[idx] * np.conj(density_matrix[idx_conj]))

    return rho_ij


def mutual_info_full_n_sites(density_matrix, trunc, n):
    """
    Calculates the mutual information of a given density matrix for 3 modes
    :param density_matrix:
    :param trunc:
    :param n: Number of modes
    :return:
    """

    rho_list = one_mode_rdm_n_sites(density_matrix, trunc, n)
    mi_list = []
    for i in range(n):
        for j in range(i+1, n):
            rho_i = rho_list[i]
            rho_j = rho_list[j]
            rho_ij = two_mode_rdm_n_sites(density_matrix, i, j, trunc, n)
            mi_list.append(mutual_information(rho_i, rho_j, rho_ij))

    return mi_list
