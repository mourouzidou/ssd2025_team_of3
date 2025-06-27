import numpy as np

class LoadDataNP:
    """
    A class to load and process data from a CSV file.
    This class provides methods to load data, filter columns by variance, 
    in numpy  
    """

    def __init__(self, path):
        self.file_path = path

    def load_data(self):
        try:
            data = np.loadtxt(self.file_path, skiprows=1)
            self.time= data[:, 0]
            self.data = data[:, 1:].T
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def filter_data(self, threshold):

        if self.data is None:
            return None
        indices = np.nonzero(np.var(self.data, axis=1) > threshold)
        print(indices)
        self.fitered_data = self.data[indices]


    