import unittest
import pandas as pd
from MultiTrain.classification.classification_models import MultiClassifier
from MultiTrain.errors.errors import (
    MultiTrainDatasetTypeError,
    MultiTrainColumnMissingError,
    MultiTrainEncodingError,
    MultiTrainTypeError,
    MultiTrainNaNError,
    MultiTrainMetricError,
    MultiTrainSplitError
)

class TestMultiClassifier(unittest.TestCase):

    def setUp(self):
        # Sample dataset for testing
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        self.target = 'target'
        self.classifier = MultiClassifier()

    def test_split_normal(self):
        # Test normal split operation
        X_train, X_test, y_train, y_test = self.classifier.split(self.data, self.target)
        self.assertEqual(len(X_train), 4)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 1)

    def test_split_with_drop(self):
        # Test split with dropping a column
        X_train, X_test, y_train, y_test = self.classifier.split(self.data, self.target, drop=['feature1'])
        self.assertNotIn('feature1', X_train.columns)
        self.assertNotIn('feature1', X_test.columns)

    def test_split_invalid_drop_type(self):
        # Test split with invalid drop type
        with self.assertRaises(MultiTrainTypeError):
            self.classifier.split(self.data, self.target, drop='feature1')

    def test_split_missing_target(self):
        # Test split with missing target column
        with self.assertRaises(MultiTrainColumnMissingError):
            self.classifier.split(self.data.drop(columns=[self.target]), self.target)

    def test_split_auto_cat_encode(self):
        # Test split with automatic categorical encoding
        X_train, X_test, y_train, y_test = self.classifier.split(self.data, self.target, auto_cat_encode=True)
        self.assertTrue(X_train['feature2'].dtype in [int, float])

    def test_fit_normal(self):
        # Test normal fit operation
        datasplits = self.classifier.split(self.data, self.target)
        results = self.classifier.fit(datasplits)
        self.assertIsInstance(results, pd.DataFrame)

    def test_fit_invalid_datasplits(self):
        # Test fit with invalid datasplits
        with self.assertRaises(MultiTrainSplitError):
            self.classifier.fit((self.data, self.data, self.data))

    def test_fit_custom_metric(self):
        # Test fit with a custom metric
        datasplits = self.classifier.split(self.data, self.target)
        with self.assertRaises(MultiTrainMetricError):
            self.classifier.fit(datasplits, custom_metric='invalid_metric')

    def test_fit_imbalanced(self):
        # Test fit with imbalanced flag
        datasplits = self.classifier.split(self.data, self.target)
        results = self.classifier.fit(datasplits, imbalanced=True)
        self.assertIsInstance(results, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()