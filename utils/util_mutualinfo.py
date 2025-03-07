# These functions are for 3 modes systems.
import numpy as np


def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # remove small rigenvalues
    return -np.sum(eigenvalues * np.log(eigenvalues))


def two_mode_rdm(f1, i, j, n):

    rho_ij = np.zeros((n ** 2, n ** 2), dtype=complex)

    # Determine which mode is being traced out
    trace_out_mode = 3 - (i + j)

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    # Using `np.sum` to trace out the `trace_out_mode` mode
                    if trace_out_mode == 0:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[:, a, b] * np.conj(f1[:, c, d]))
                    elif trace_out_mode == 1:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[a, :, b] * np.conj(f1[c, :, d]))
                    elif trace_out_mode == 2:
                        rho_ij[a * n + b, c * n + d] = np.sum(
                            f1[a, b, :] * np.conj(f1[c, d, :]))

    return rho_ij


def mutual_information(rho_i, rho_j, rho_ij):
    s_i = von_neumann_entropy(rho_i)
    s_j = von_neumann_entropy(rho_j)
    s_ij = von_neumann_entropy(rho_ij)
    return s_i + s_j - s_ij


# single-mode RDMs

def one_mode_rdm(f1, trunc):

    rho_1 = np.zeros((trunc, trunc), dtype=complex)
    rho_2 = np.zeros((trunc, trunc), dtype=complex)
    rho_3 = np.zeros((trunc, trunc), dtype=complex)

    for i in range(trunc):
        for j in range(trunc):
            rho_1[i, j] = np.sum(f1[i, :, :] * np.conj(f1[j, :, :]))
            rho_2[i, j] = np.sum(f1[:, i, :] * np.conj(f1[:, j, :]))
            rho_3[i, j] = np.sum(f1[:, :, i] * np.conj(f1[:, :, j]))

    return rho_1, rho_2, rho_3



def mutual_info_full(f1, trunc):

    rho_1, rho_2, rho_3 = one_mode_rdm(f1, trunc)
    rho_12 = two_mode_rdm(f1, 0, 1, trunc)
    rho_13 = two_mode_rdm(f1, 0, 2, trunc)
    rho_23 = two_mode_rdm(f1, 1, 2, trunc)

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


def mutual_info_cost(f1, n, fn):

    s1, s2, s3 = single_entropy_full(f1, n)

    return fn(s1, s2, s3)
