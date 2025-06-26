import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# create a class ProcessData to load and process data
class ProcessDataPd:
    """
    A class to load and process data from a CSV file.
    This class provides methods to load data, discard columns by index,
    filter columns by variance, create the upper triangle of the correlation matrix,
    and get sorted correlations from the correlation matrix.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
    
    def load_data(self, index_col=None):
        """Load data from a CSV file with an optional index column."""
        self.data = pd.read_csv(self.file_path, index_col=index_col)
        return self.data

    def discard_columns_by_index(self, indices: list):
        """
        Discard columns by their indices.
        e.g. remove columns with constant values 
        [0, 1, 2] will remove the first three columns.
        """
        if self.data is not None:
            self.data.drop(self.data.columns[indices], axis=1, inplace=True)
        return self.data
    
    def filter_by_variance(self, threshold: float):
        """
        Filter columns by variance.
        e.g. remove columns with variance below the threshold.
        """
        if self.data is not None:
            variances = self.data.var()
            self.data = self.data.loc[:, variances > threshold]
        return self.data
    
    def upper_correlation_triangle(self):
        """
        Create the upper triangle of the correlation matrix.
        """
        if self.data is not None:
            corr = self.data.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            upper_corr = corr.where(mask)
            return upper_corr
        return None
    


    def get_sorted_correlations(correlation_matrix, save_to_txt=False):
        """
        Get sorted correlations from the correlation matrix.
        
        Parameters:
        correlation_matrix (DataFrame): The correlation matrix.
        
        Returns:
        Series: Sorted correlations with absolute values.
        

        """
        upper_triangular = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        
        # Flatten and sort by absolute values
        sorted_correlations = upper_triangular.stack().abs().sort_values(ascending=False)
        
        return sorted_correlations



