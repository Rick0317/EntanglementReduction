import unittest
from RickVersion.utils.util_gfro import *
from RickVersion.utils.util_hamil import bilinear_three_mode_H, anharmonic_three_mode_H
from RickVersion.utils.util_mutualinfo import mutual_info_full
from RickVersion.Graph_plots.mutual_information_visualizer import visualize_mutual_information
import numpy as np


class TestModeSwap(unittest.TestCase):
    def test_mode_swap(self):
        h_variables = [1 / 2, 1 / 2, 1 / 2, 0.1, 0.2, 0.3]
        model_H = bilinear_three_mode_H(h_variables)
        truncation = 10
        eig_value, eig_vec = boson_eigenspectrum_sparse(model_H, truncation, 1)

        ed_ground_state = eig_vec
        print(f"Prior ground state energy: {eig_value}")


        reshaped_gs = ed_ground_state.reshape(truncation, truncation,
                                              truncation)
        I_12, I_23, I_13 = mutual_info_full(reshaped_gs, truncation)
        visualize_mutual_information(I_12, I_23, I_13)

        mode_swapped_H = mode_swap(model_H, 1, 2)
        eig_value2, eig_vec = boson_eigenspectrum_sparse(mode_swapped_H, truncation, 1)

        ed_ground_state = eig_vec

        reshaped_gs = ed_ground_state.reshape(truncation, truncation,
                                              truncation)
        I_12, I_23, I_13 = mutual_info_full(reshaped_gs, truncation)
        visualize_mutual_information(I_12, I_23, I_13)

        assert np.isclose(eig_value, eig_value2)
