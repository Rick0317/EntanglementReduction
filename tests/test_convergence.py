"""
Test the convergence of the ground-state with a given truncation
"""

from utils.util_gfro import boson_eigenspectrum_sparse
from utils.util_hamil import bilinear_three_mode_H, anharmonic_three_mode_H


def get_Hamiltonian():
    h_variables = [1 / 2, 1 / 2, 1 / 2, 0.6]
    model_H = anharmonic_three_mode_H(h_variables)
    return model_H


if __name__ == '__main__':

    truncation = 10
    model_H = get_Hamiltonian()

    eig_value, _ = boson_eigenspectrum_sparse(model_H, truncation, 1)

    truncation2 = 11
    model_H = get_Hamiltonian()

    eig_value2, _ = boson_eigenspectrum_sparse(model_H, truncation, 1)

    print(f"Truncations: {truncation}, {truncation2}")
    print(f"Eigenvalue difference: {abs(eig_value-eig_value2)}")
