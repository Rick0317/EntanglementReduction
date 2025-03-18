"""
Test the implementation of Bogoliubov transformation.
"""

import unittest

import numpy as np


class TestMutualInformation(unittest.TestCase):

    def test_initial_PQ(self):
        n = 3
        P, Q = initial_only_PQ(n)

        P_dg = P.conj().T
        Q_dg = Q.conj().T

        assert np.allclose(-P_dg, P), "P should be skew symmetric"
        assert np.allclose(Q_dg, Q), "Q should be symmetric"

    def test_u_v_construction(self):
        n = 3
        P, Q = initial_only_PQ(n)
        expA = super_matrix_exp(P, Q)
        U, V = extract_sub_matrices(expA, n)

        relation1 = U.conj().T @ U - V.conj().T @ V
        relation2 = U.conj().T @ V - V.conj().T @ U

        assert np.allclose(relation1, np.eye(n)), "Relation1 is incorrect"
        assert np.allclose(relation2, np.zeros((n, n))), "Relation2 is incorrect"

    def test_parameters_construction(self):
        n = 3
        P, Q = initial_only_PQ(n)
        X = 1e-6 * flatten_matrices_only_PQ(P, Q, n)
        P, Q = recreate_matrices_PQ_only(X, n)

        expA = super_matrix_exp(P, Q)
        U, V = extract_sub_matrices(expA, n)

        relation1 = U.conj().T @ U - V.conj().T @ V
        relation2 = U.conj().T @ V - V.conj().T @ U

        assert np.allclose(relation1, np.eye(n)), "Relation1 is incorrect"
        assert np.allclose(relation2,
                           np.zeros((n, n))), "Relation2 is incorrect"


    def test_transformed_op_inv_no_translation(self):
        n = 3
        P, Q = initial_only_PQ(n)

        expA = super_matrix_exp(P, Q)
        U, V = extract_sub_matrices(expA, n)

        B_b, B_bd = transformed_op_inv_no_translation(U, V)

        assert len(B_b) == len(B_bd), "Length of B is incorrect"
        assert len(B_b) == n, "Length of B is incorrect"

    def test_transformed_op_inv_no_translation_custom_UV(self):
        n = 2
        U = np.array([[1, 0], [0, -1]])
        V = np.array([[1, 1], [1, -1]])

        B_b, B_bd = transformed_op_inv_no_translation(U, V)

        assert len(B_b) == len(B_bd), "Length of B is incorrect"
        assert len(B_b) == n, "Length of B is incorrect"

