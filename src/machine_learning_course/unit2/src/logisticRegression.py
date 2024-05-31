import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogisticRegression():
    def __init__(self, alpha=0.1, numIterations=100000):
        self.alpha = alpha
        self.numIterations = numIterations
        self.logisticLoss = []
    
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):
       return 1 / (1 + np.exp(-z))
    
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])

        for i in range(self.numIterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h-y)) / y.size
            self.theta -= self.alpha * gradient

            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            loss = self.loss(h, y)

            if( i % 1000 == 0 ):
                self.logisticLoss.append(loss)
            if( i % 5000 == 0 ):
                print("i:", i, "loss: ", loss)
    
    def predictProb(self, X):
        return self.sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predictProb(X).round()

# Load the DataSet
data = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/test_logistic_data.csv')
X = np.asarray(data.iloc[:, :-1])
y = np.asarray(data.iloc[:, -1])

# Creating LogisticRegression Object
model = LogisticRegression(alpha=0.1, numIterations=100000)
model.fit(X, y)

# Print the Error During Training
plt.figure(figsize=(10, 8))
i = np.arange(0, len(model.logisticLoss),1)
plt.plot(i, model.logisticLoss)
plt.xlabel("Iteration", fontsize=16)
plt.ylabel("Error", fontsize=16)
plt.show()

# Show the Logistic Regression
plt.figure(figsize=(10, 8))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1],  label='1')
plt.legend()
x1_min, x1_max = X[:,0].min(), X[:,0].max(),
x2_min, x2_max = X[:,1].min(), X[:,1].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predictProb(grid).reshape(xx1.shape)
plt.xlabel("Deviation - x direction", fontsize=16)
plt.ylabel("Deviation - y direction", fontsize=16)
plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='red')
plt.show()
