import numpy as np

class FourrierTranformData:
    def __init__(self, data):
        self.data = data
        self.fft_data = None
        

    def do_DFT(self ):
        self.data_fft = np.fft.rfft(self.data)
        self.frequencies = np.fft.rfftfreq(len(self.data), d=1.0)
    
