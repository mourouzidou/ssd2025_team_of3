# import unittest
import pytest
import create_plots as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# class TestCreatePlots(unittest.TestCase):
#     def setUp(self):
#         # Sample data for testing
#         self.data = pd.DataFrame({
#             'A': [1, 2, 3, 4, 5],
#             'B': [5, 4, 3, 2, 1],
#             'C': [2, 3, 4, 5, 6]
#         })
#         self.plotter = cp.PlotData(self.data)

#     def test_plot_correlation_matrix(self):
#         """Test if the correlation matrix plot can be created."""
#         try:
#             self.plotter.plot_correlation_matrix()
#             plt.close()  # Close the plot to avoid displaying it during tests
#         except Exception as e:
#             self.fail(f"plot_correlation_matrix raised an exception: {e}")

@pytest.fixture
def setup(tmp_path):
    # Sample data for testing
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6]
    })
    plotter = cp.PlotData(data)
    return plotter

def test_plot_correlation_matrix(setup, tmp_path):
    """Test if the correlation matrix plot can be created."""
    save_path = tmp_path / "correlation_matrix.png"
    setup.plot_correlation_matrix(save_path=save_path)
    assert save_path.exists
    save_path.unlink  # Clean up the saved file after test
    setup.plot_correlation_matrix(save_path=None)
    with pytest.raises(ValueError):
        setup.data = None  # Simulate no data available
        setup.plot_correlation_matrix()
    