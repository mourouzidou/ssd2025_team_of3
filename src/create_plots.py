import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

class PlotData:
    def __init__(self, data):
        self.data = data

    def plot_correlation_matrix(self, save_path=None):
        """
        Plot the correlation matrix of the data.
        If save_path is provided, save the plot to that path.
        """
        if self.data is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.data.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
            plt.title('Correlation Matrix')
            if save_path:
                plt.savefig(save_path)
            plt.show()
        else:
            print("No data available to plot.")

    def plot_histogram(self, column, bins=30, save_path=None):
        """
        Plot a histogram of a specified column in the data.
        If save_path is provided, save the plot to that path.
        """
        if self.data is not None and column in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[column], bins=bins, kde=True)
            plt.title(f'Histogram of {column}')
            if save_path:
                plt.savefig(save_path)
            plt.show()
        else:
            print(f"No data available for column: {column}")




# create a class CreatePlots to create plots