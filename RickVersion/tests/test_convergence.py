"""
Test the convergence of the ground-state with a given truncation
"""

from RickVersion.utils.util_gfro import boson_eigenspectrum_sparse
from RickVersion.utils.util_hamil import anharmonic_three_mode_H


def get_Hamiltonian():
    h_variables = [1 / 2, 1 / 2, 1 / 2, 0.6, 0.6, 0.6]
    model_H = anharmonic_three_mode_H(h_variables)
    return model_H


if __name__ == '__main__':

    truncation = 6
    model_H = get_Hamiltonian()

    eig_value, _ = boson_eigenspectrum_sparse(model_H, truncation, 1)

    truncation2 = 7
    model_H = get_Hamiltonian()

    eig_value2, _ = boson_eigenspectrum_sparse(model_H, truncation, 1)

    print(f"Truncations: {truncation}, {truncation2}")
    print(f"Eigenvalue difference: {abs(eig_value-eig_value2)}")
