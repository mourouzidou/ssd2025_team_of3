import unittest
import pandas as pd
import numpy as np
import tempfile
import os


class TestProcessDataPd(unittest.TestCase):
    
    def setUp(self):
        # Create simple test data
        self.test_data = pd.DataFrame({
            'A': [1, 2, 3, 4], 
            'B': [2, 4, 6, 8],
            'C': [1, 1, 1, 1],  # constant column
            'D': [5, 6, 7, 8]
        })
        
        # Save to temp file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
        
        self.processor = ProcessDataPd(self.temp_file.name)
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def test_load_data(self):
        data = self.processor.load_data()
        self.assertEqual(len(data), 4)
        self.assertEqual(len(data.columns), 4)
    
    def test_discard_columns(self):
        self.processor.load_data()
        result = self.processor.discard_columns_by_index([2])  # Remove column C
        self.assertNotIn('C', result.columns)
        self.assertEqual(len(result.columns), 3)
    
    def test_filter_by_variance(self):
        self.processor.load_data()
        result = self.processor.filter_by_variance(0.1)  # Remove constant columns
        self.assertNotIn('C', result.columns)  # C has variance 0
    
    def test_correlation_matrix(self):
        self.processor.load_data()
        corr = self.processor.upper_correlation_triangle()
        self.assertIsNotNone(corr)
        # Check diagonal is 1.0
        self.assertEqual(corr.iloc[0, 0], 1.0)
        # Check lower triangle is NaN
        self.assertTrue(pd.isna(corr.iloc[1, 0]))

def test_get_sorted_correlations():
    # Simple correlation matrix
    corr = pd.DataFrame({
        'A': [1.0, 0.8, 0.3],
        'B': [0.8, 1.0, 0.6],
        'C': [0.3, 0.6, 1.0]
    }, index=['A', 'B', 'C'])
    
    result = get_sorted_correlations(corr)
    # Highest correlation should be first
    assert result.iloc[0] == 0.8

if __name__ == '__main__':
    unittest.main()