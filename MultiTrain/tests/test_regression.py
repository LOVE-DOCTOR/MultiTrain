"""
test cases for Classification class
"""

import unittest

# import MultiTrain
from MultiTrain.classification import classification_models

X = [[1, 0], [0, 1], [1, 1]]
Y1 = [0.3, 2.3, 1.8]


class testClassification(unittest.TestCase):
    """
    Test Regression methods
    """

    def test_fit(self):
        pass
