import unittest

import utils.util_mutualinfo
import numpy as np


class TestMutualInformation(unittest.TestCase):
    def test_von_neumann_entropy(self):
        """
        Test the von neumann entropy function.
        :return:
        """
