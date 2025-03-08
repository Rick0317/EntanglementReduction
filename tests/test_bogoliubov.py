"""
Test the implementation of Bogoliubov transformation.
"""

import unittest

from utils.util_gfro import *
import numpy as np


class TestMutualInformation(unittest.TestCase):
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

