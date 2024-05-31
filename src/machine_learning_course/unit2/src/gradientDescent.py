from __future__ import division
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import future 
import warnings
warnings.filterwarnings('ignore')

class GradientDescent(object):
    def __init__(self):
        self._MSE = []
        self._iteration = []
        self._data = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/test_brakes.csv')

        # Clean Up Data
        self._data.drop(['Unnamed: 0'], axis = 1, inplace = True)
        self._data.drop(['payload'], axis = 1, inplace = True)
        
        self._x = self._data.iloc[:,1].values.flatten()
        self._y = self._data.iloc[:,0].values.flatten()

    def getX(self):
        return self._x
    
    def getY(self):
        return self._y
    
    def computeErrorForLineGivenPoints(self,b,m,X,Y):
        totalError = 0
        for i in range(0, len(X)):
            x = X
            y = Y
            functionResult = m*x[i] + b
            totalError += (y[i]-functionResult)**2
        return (totalError)/len(X)
    
if __name__ == "__main__":
    # Running our Gradient Descent
    gd = GradientDescent()
    m = 0
    b = 0
    alpha = 0.0001
    epochs = 15000
    n = len(gd.getX())

    for i in range(epochs):
        # Get the Gradient Values
        Y_pred = m*gd.getX() + b
        gradientM = (-2/n) * sum(gd.getX() * (gd.getY() - Y_pred))
        gradientB = (-2/n) * sum(gd.getY() - Y_pred)

        # Update m and b
        m = m - alpha * gradientM
        b = b - alpha * gradientB
        
        # Get the new MSE values
        mse = gd.computeErrorForLineGivenPoints(b, m, gd._x, gd._y)
        gd._MSE.append(mse)
        gd._iteration.append(i)
    
    # Plot our final m and b values
    Y_pred = m * gd.getX() + b

    plt.figure(figsize=(10,8))
    plt.scatter(gd.getX(),gd.getY())
    plt.plot([min(gd.getX()),max(gd.getX())], [min(Y_pred),max(Y_pred)], color='red')

    plt.xlabel("% of max speed of axis 1", fontsize=16)
    plt.ylabel("Stop Distance [Degress]", fontsize=16)
    plt.title("Gradient Descent", fontsize=18)
    
        
    print ("line parameters :::",  "m: ", m, "b: ", b)
    print ("MSE for gradient descent :::", gd._MSE[-1])

    plt.figure(figsize=(10,8))
    i = np.arange(0,len(gd._MSE), 1)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.title('Gradient descent ERROR (MSE)', fontsize=18)
    plt.plot(i,gd._MSE)
    plt.show()



