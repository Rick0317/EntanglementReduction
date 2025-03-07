# These functions are for 3d models

import numpy as np


def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # remove small rigenvalues
    return -np.sum(eigenvalues * np.log(eigenvalues))


def two_mode_rdm(f1, i, j, n):

    rho_ij = np.zeros((n ** 2, n ** 2), dtype=complex)

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    # Using `np.sum` to trace out the `trace_out_mode` mode
                    if i == 0 and j == 1:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[a, b, :, :] * np.conj(f1[c, d, :, :]))
                    elif i == 0 and j == 2:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[a, :, b, :] * np.conj(f1[c, :, d, :]))
                    elif i == 0 and j == 3:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[a, :, :, b] * np.conj(f1[c, :, :, d]))

                    elif i == 1 and j == 2:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[:, a, b, :] * np.conj(f1[:, c, d, :]))

                    elif i == 1 and j == 3:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[:, a, :, b] * np.conj(f1[:, c, :, d]))

                    elif i == 2 and j == 3:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[:, :, a, b] * np.conj(f1[:, :, c, d]))

    return rho_ij


def two_mode_rdm3(f1, i, j, n):
    # Initialize the reduced density matrix (RDM)
    rho_ij = np.zeros((n ** 2, n ** 2), dtype=complex)

    # Determine which mode is being traced out
    trace_out_mode = 3 - (i + j)

    if trace_out_mode == 0:
        # Sum over the first index of `f1`
        rho_ij = np.einsum('kab,kcd->abcd', f1, np.conj(f1)).reshape(n ** 2,
                                                                     n ** 2)
    elif trace_out_mode == 1:
        # Sum over the second index of `f1`
        rho_ij = np.einsum('akb,ckd->abcd', f1, np.conj(f1)).reshape(n ** 2,
                                                                     n ** 2)
    elif trace_out_mode == 2:
        # Sum over the third index of `f1`
        rho_ij = np.einsum('abk,cdk->abcd', f1, np.conj(f1)).reshape(n ** 2,
                                                                     n ** 2)

    return rho_ij


def mutual_information(rho_i, rho_j, rho_ij):
    s_i = von_neumann_entropy(rho_i)
    s_j = von_neumann_entropy(rho_j)
    s_ij = von_neumann_entropy(rho_ij)
    return s_i + s_j - s_ij


# single-mode RDMs

def one_mode_rdm(f1, n):

    rho_1 = np.zeros((n, n), dtype=complex)
    rho_2 = np.zeros((n, n), dtype=complex)
    rho_3 = np.zeros((n, n), dtype=complex)
    rho_4 = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            rho_1[i, j] = np.sum(f1[i, :, :, :] * np.conj(f1[j, :, :, :]))
            rho_2[i, j] = np.sum(f1[:, i, :, :] * np.conj(f1[:, j, :, :]))
            rho_3[i, j] = np.sum(f1[:, :, i, :] * np.conj(f1[:, :, j, :]))
            rho_4[i, j] = np.sum(f1[:, :, :, i] * np.conj(f1[:, :, :, j]))

    return rho_1, rho_2, rho_3, rho_4



def mutual_info_full(f1, n):

    rho_1, rho_2, rho_3, rho_4 = one_mode_rdm(f1, n)
    rho_12 = two_mode_rdm(f1, 0, 1, n)
    rho_13 = two_mode_rdm(f1, 0, 2, n)
    rho_14 = two_mode_rdm(f1, 0, 3, n)
    rho_23 = two_mode_rdm(f1, 1, 2, n)
    rho_24 = two_mode_rdm(f1, 1, 3, n)
    rho_34 = two_mode_rdm(f1, 2, 3, n)


    I_12 = mutual_information(rho_1, rho_2, rho_12)
    I_13 = mutual_information(rho_1, rho_3, rho_13)
    I_14 = mutual_information(rho_1, rho_4, rho_14)
    I_23 = mutual_information(rho_2, rho_3, rho_23)
    I_24 = mutual_information(rho_2, rho_4, rho_24)
    I_34 = mutual_information(rho_3, rho_4, rho_34)

    return I_12, I_13, I_14, I_23, I_24, I_34


def print_mutual_all(f1, n):

    rho_1, rho_2, rho_3 = one_mode_rdm(f1, n)
    rho_12 = two_mode_rdm(f1, 0, 1, n)
    rho_13 = two_mode_rdm(f1, 0, 2, n)
    rho_23 = two_mode_rdm(f1, 1, 2, n)

    s1 = von_neumann_entropy(rho_1)
    s2 = von_neumann_entropy(rho_2)
    s3 = von_neumann_entropy(rho_3)

    s12 = von_neumann_entropy(rho_12)
    s23 = von_neumann_entropy(rho_23)
    s13 = von_neumann_entropy(rho_13)

    I_12, I_23, I_13 = mutual_info_full(f1, n)

    print(
        f"\n {np.round(I_12, 7)}, {np.round(I_23, 7)}, {np.round(I_13, 7)}, "
        f"{np.round(s1, 7)}, {np.round(s2, 7)}, {np.round(s3, 7)}, "
        f"{np.round(s12, 7)}, {np.round(s23, 7)}, {np.round(s13, 7)}"
    )

    return


def mutual_all(f1, n):

    I_12, I_23, I_13 = mutual_info_full(f1, n)

    return I_12, I_23, I_13


def single_entropy_full(f1, n):

    rho_1, rho_2, rho_3 = one_mode_rdm(f1, n)

    s1 = von_neumann_entropy(rho_1)
    s2 = von_neumann_entropy(rho_2)
    s3 = von_neumann_entropy(rho_3)

    return s1, s2, s3


def mutual_info_cost1(f1, n, fn):

    s1, s2, s3 = single_entropy_full(f1, n)

    return fn(s1, s2, s3)


def mutual_info_cost(f1, n, fn):

    I_12, I_23, I_13 = mutual_info_full(f1, n)

    return fn(I_12, I_23, I_13)
