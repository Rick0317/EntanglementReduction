import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from RickVersion.utils.util_gfro import boson_eigenspectrum_sparse
from RickVersion.utils.util_mutualinfo import mutual_info_full
from RickVersion.utils.util_hamil import bilinear_three_mode_H, anharmonic_three_mode_H
from RickVersion.utils.util_tensornetwork import get_approx_bd_mps

def get_Hamiltonian():
    h_variables = [1 / 2, 1 / 2, 1 / 2, 0.6, 0.6, 0]
    model_H = anharmonic_three_mode_H(h_variables)
    return model_H



def visualize_mutual_information(MI, n):
    """
    Visualize the mutual information between n modes.

    :param MI: List of mutual information values in upper triangular order.
    :param n: Number of modes.
    """
    # Initialize an empty mutual information matrix
    mutual_info_matrix = np.zeros((n, n))

    # Fill the upper triangular part of the matrix
    index = 0
    for i in range(n):
        for j in range(i + 1, n):
            mutual_info_matrix[i, j] = MI[index]
            mutual_info_matrix[j, i] = MI[index]  # Symmetric assignment
            index += 1

    # Plot the heatmap
    sns.heatmap(mutual_info_matrix, annot=True, fmt=".5f", cmap="Blues",
                xticklabels=[str(i + 1) for i in range(n)],
                yticklabels=[str(i + 1) for i in range(n)])
    plt.title("Mutual Information Heatmap")
    plt.show()



if __name__ == '__main__':
    model = "anharmonic_three_mode_H"
    truncation = 10
    model_H = get_Hamiltonian()

    eig_value, eig_vec = boson_eigenspectrum_sparse(model_H, truncation, 1)

    ed_ground_state = eig_vec

    ed_ground_state_energy = eig_value

    reshaped_gs = ed_ground_state.reshape(truncation, truncation, truncation)

    I_12, I_23, I_13 = mutual_info_full(reshaped_gs, truncation)
    visualize_mutual_information(I_12, I_23, I_13)

    error_threshold_frac = 1e-4
    threshold = abs(ed_ground_state_energy * error_threshold_frac)
    exact_bd = get_approx_bd_mps(reshaped_gs, threshold=threshold)
    print(f"Almost Exact BD {threshold}: {exact_bd}")
    print(f"Ground state energy: {ed_ground_state_energy}")
