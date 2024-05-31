
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('/home/user/catkin_ws/src/results/test.csv')
MSE = []
iteration = []

X = np.asarray(data.iloc[:, 0])
Y = np.asarray(data.iloc[:, 1])

def computeErrorForLineGivenPoints(b,m,X,Y):
        totalError = 0
        for i in range(0, len(X)):
            x = X
            y = Y
            functionResult = m*x[i] + b
            totalError += (y[i]-functionResult)**2
        return (totalError)/len(X)

# Running our Gradient Descent
m = 0
b = 0
alpha = 0.0001
epochs = 15000
n = len(X)

for i in range(epochs):
    # Get the Gradient Values
    Y_pred = m*X + b
    gradientM = (-2/n) * sum(X * (Y - Y_pred))
    gradientB = (-2/n) * sum(Y - Y_pred)

    # Update m and b
    m = m - alpha * gradientM
    b = b - alpha * gradientB
    
    # Get the new MSE values
    mse = computeErrorForLineGivenPoints(b, m, X, Y)
    MSE.append(mse)
    iteration.append(i)

# Plot our final m and b values
Y_pred = m * X + b
X_future = 1.4
Y_future = m * X_future + b

plt.figure(figsize=(10, 10))
plt.scatter(X, Y, label = 'ground truth')
plt.scatter(X_future,Y_future, label = 'prediction')
plt.plot(X, Y_pred, color='red',label = 'prediction' )

plt.xlabel("X", fontsize=16)
plt.ylabel("Y", fontsize=16)
plt.title('Gradient descent', fontsize=18)
plt.legend()
plt.show()

# plt.figure(figsize=(10, 10))
# plt.scatter(X,y)
# plt.xlabel("X", fontsize=16)
# plt.ylabel("Y", fontsize=16)
# plt.title("Robot position", fontsize=18)

# plt.show()