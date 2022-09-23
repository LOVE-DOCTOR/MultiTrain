"""
test cases for Classification class
"""

import unittest
from MultiTrain.classification.classification_models import MultiClassifier

X = [[1, 0], [0, 1], [1, 1]]
Y1 = [0, 1, 1]


class testClassification(unittest.TestCase):
    """
    Test Classification methods
    """

    def test_fit(self):
        pass
