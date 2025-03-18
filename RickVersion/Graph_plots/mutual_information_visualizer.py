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



def visualize_mutual_information(I_12, I_23, I_13):
    """
    Visualize the mutual information between three modes.
    :param I_12:
    :param I_23:
    :param I_13:
    :return:
    """
    mutual_info_matrix = np.array([[0, I_12, I_13],
                                   [I_12, 0, I_23],
                                   [I_13, I_23, 0]])

    # Plot the heatmap
    sns.heatmap(mutual_info_matrix, annot=True, fmt=".5f", cmap="Blues",
                xticklabels=["1", "2", "3"], yticklabels=["1", "2", "3"])
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
