"""
Test the implementation of Mutual Information.
"""

import unittest

from RickVersion.utils.util_mutualinfo import von_neumann_entropy, mutual_info_full, one_mode_rdm
import numpy as np


class TestMutualInformation(unittest.TestCase):
    def test_von_neumann_entropy_1_qubit_mixed(self):
        """
        Test the von neumann entropy function for a mixed state.
        :return:
        """
        rho = np.array([[0.5, 0], [0, 0.5]])
        expect = - 2 * 0.5 * np.log(0.5)
        entropy = von_neumann_entropy(rho)

        assert entropy == expect, "von neumann entropy failed"

    def test_von_neumann_entropy_1_qubit_pure(self):
        """
        Test the von neumann entropy function for a pure state.
        :return:
        """
        rho = np.array([[0.5, 0.5], [0.5, 0.5]])
        expect = 0
        entropy = von_neumann_entropy(rho)

        assert entropy == expect, "von neumann entropy failed"

    def test_one_mode_rdm(self):
        """
        Test the one mode RDM function.
        :return:
        """
        trunc = 2
        # w_state: (trunc, trunc, trunc) density matrix
        w_state = np.array(
            [0, 1 / np.sqrt(3), 1 / np.sqrt(3), 0, 1 / np.sqrt(3), 0, 0, 0])
        reshaped_w_state = w_state.reshape(trunc, trunc, trunc)
        rho_1, rho_2, rho_3 = one_mode_rdm(reshaped_w_state, trunc)
        expected_rho1 = np.array([[2/3, 0], [0, 1/3]])
        assert np.allclose(rho_1, expected_rho1), "one mode RDM failed"

    def test_mutual_information_w_state(self):
        """
        Test the mutual_info_full function with W state.
        :return:
        """
        trunc = 2
        w_state = np.array([0, 1/np.sqrt(3), 1/np.sqrt(3), 0, 1/np.sqrt(3), 0, 0, 0])
        reshaped_w_state = w_state.reshape(trunc, trunc, trunc)
        expected_Ixy = - 1 / 3 * np.log(1/3) - 2/3 * np.log(2/3)
        Ixy, Iyz, Izx = mutual_info_full(reshaped_w_state, trunc)
        assert Ixy == expected_Ixy, "Mutual Information failed"
        assert Ixy == Iyz, "Mutual Information failed"
        assert Ixy == Izx, "Mutual Information failed"

    def test_mutual_information_state2(self):
        """
        Test the mutual_info_full function with another state
        :return:
        """
        trunc = 2
        w_state = np.array(
            [0, 1 / np.sqrt(3), 0, 0, 0, 0, 1 / np.sqrt(3), 1 / np.sqrt(3)])
        reshaped_w_state = w_state.reshape(trunc, trunc, trunc)

        eig_val1 = (3 - np.sqrt(5)) / 6
        eig_val2 = (3 + np.sqrt(5)) / 6

        s_x = - 1 / 3 * np.log(1 / 3) - 2 / 3 * np.log(2 / 3)
        s_y = - 1 / 3 * np.log(1 / 3) - 2 / 3 * np.log(2 / 3)
        s_z = - eig_val1 * np.log(eig_val1) - eig_val2 * np.log(eig_val2)
        s_xy = - eig_val1 * np.log(eig_val1) - eig_val2 * np.log(eig_val2)
        s_yz = - 1 / 3 * np.log(1 / 3) - 2 / 3 * np.log(2 / 3)
        s_xz = - 1 / 3 * np.log(1 / 3) - 2 / 3 * np.log(2 / 3)

        expected_Ixy = s_x + s_y - s_xy
        expected_Iyz = s_y + s_z - s_yz
        expected_Izx = s_z + s_x - s_xz
        Ixy, Iyz, Izx = mutual_info_full(reshaped_w_state, trunc)
        assert Ixy == expected_Ixy, "Mutual Information failed"
        assert Iyz == expected_Iyz, "Mutual Information failed"
        assert Izx == expected_Izx, "Mutual Information failed"
