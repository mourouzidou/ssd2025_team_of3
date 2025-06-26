import numpy as np

class LoadDataNP:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        try:
            data = np.load(self.file_path, skiprows=1)
            self.time= data[:, 0]
            self.data = data[:, 1:].T
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def filter_data(self, threshold):

        if self.data is None:
            return None
        indices = np.nonzero(np.var(self.data, axis=1) > threshold)
        self.fitered_data = self.data[np.indices]


    