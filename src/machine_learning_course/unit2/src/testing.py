
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

class LinearRegression(object):
    def __init__(self):
        self._m = 0
        self._b = 0

    def m(self):
        return self._m
    
    def b(self):
        return self._b
    
    def fit(self, X, y):
        pass
    
    def predict(self, x):
        pass
    
def MSE(ax, x, y, model):
    pass

def compute_regression(ax, x, y, model):
    pass

model = LinearRegression()

data = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/test_brakes.csv')

data.drop(['Unnamed: 0'], axis=1, inplace=True)

yA = (data.iloc[:,0].values).flatten()
x = (data.iloc[:,1].values).flatten()