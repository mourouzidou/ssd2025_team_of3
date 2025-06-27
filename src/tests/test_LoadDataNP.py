import LoadDataNP as LD
import pytest
import numpy as np

@pytest.fixture
def my_data():
    data= LD.LoadDataNP("tests/test_data.csv")
    return data

def test_load_data_time(my_data):
    my_data.load_data()
    assert np.all(my_data.time == [0.000000000,    0.100000000 ,   0.200000000,    0.300000000,
    0.400000000,    0.500000000,    0.600000000,    0.700000000,
    0.800000000])
    
def test_load_data_data(my_data):
    my_data.load_data()
    expected_data= np.zeros([3,9])
    assert np.all(my_data.data == expected_data)
    
def test_filter_data(my_data):
    my_data.load_data()
    my_data.filter_data(0.1)
    
    assert len(my_data.fitered_data) == 0
    