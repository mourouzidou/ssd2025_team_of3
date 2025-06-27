import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from process_data import ProcessDataPd

@pytest.fixture
def sample_data():
    """Create sample test data"""
    return pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [2, 4, 6, 8],
        'C': [1, 1, 1, 1],  # constant column
        'D': [5, 6, 7, 8]
    })

@pytest.fixture
def temp_csv_file(sample_data):
    """Create temporary CSV file with sample data"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    sample_data.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)

@pytest.fixture
def processor(temp_csv_file):
    """Create ProcessDataPd instance"""
    return ProcessDataPd(temp_csv_file)

def test_load_data(processor):
    data = processor.load_data()
    assert len(data) == 4
    assert len(data.columns) == 4
    assert processor.data is not None

def test_discard_columns(processor):
    processor.load_data()
    result = processor.discard_columns_by_index([2])  # Remove column C
    assert 'C' not in result.columns
    assert len(result.columns) == 3

def test_filter_by_variance(processor):
    processor.load_data()
    result = processor.filter_by_variance(0.1)  # Remove constant columns
    assert 'C' not in result.columns  # C has variance 0
    assert len(result.columns) == 3

def test_correlation_matrix(processor):
    processor.load_data()
    corr = processor.upper_correlation_triangle()
    assert corr is not None
    # Check diagonal is 1.0
    assert corr.iloc[0, 0] == 1.0
    # Check lower triangle is NaN
    assert pd.isna(corr.iloc[1, 0])

def test_get_sorted_correlations():
    # Simple correlation matrix
    corr = pd.DataFrame({
        'A': [1.0, 0.8, 0.3],
        'B': [0.8, 1.0, 0.6],
        'C': [0.3, 0.6, 1.0]
    }, index=['A', 'B', 'C'])
    
    result = ProcessDataPd.get_sorted_correlations(corr)
    # Highest correlation should be first
    assert result.iloc[0] == 0.8
    assert len(result) == 3  # Only upper triangle correlations

def test_no_data_handling(processor):
    # Test methods when no data is loaded
    assert processor.discard_columns_by_index([0]) is None
    assert processor.filter_by_variance(0.1) is None
    assert processor.upper_correlation_triangle() is None