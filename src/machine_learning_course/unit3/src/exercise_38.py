import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

def create_data_set(points, classes):
    
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.5
        X[ix] = np.c_[r*np.sin(t*1.0), r*np.cos(t*1.0)]
        y[ix] = class_number
    return X, y

X, y = create_data_set(400, 2)

N = 200 # Number of Points Per Class
D = 2 # Dimensionality
K = 2 # Number of Classes
h = 1000 # Size of Hidden Layer
step_size = 1e-0

# Weights are randomly initialized
W = 0.01 * np.random.randn(D, h)
W2 = 0.01 * np.random.randn(h,K)

# Biases are set to 1
b = np.zeros((1,h))
b2 = np.zeros((1,K))

num_examples = X.shape[0]
ii = []
error = []

for i in range(3000):
    # RELU Activation
    hiddenLayer = np.maximum(0, np.dot(X,W) + b)
    scores = np.dot(hiddenLayer, W2) + b2

    # SoftMax
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Cross Entrpy
    corect_logprobs = -np.log(probs[range(num_examples),y])

    # Cross Entropy Loss
    data_loss = np.sum(corect_logprobs)/num_examples
    loss = data_loss
    if i % 100 == 0:
        print("i:", i, "loss: ", loss)
        ii.append(i)
        error.append(loss)

    # Gradient on Scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples

    # Backpropagation
    dW2 = np.dot(hiddenLayer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next BP into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hiddenLayer <= 0] = 0
    # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
  
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2


# # Utilizing SKLearn
# # Define your model
# model = MLPClassifier(hidden_layer_sizes=(1000,), activation='relu', solver='sgd', alpha=0, batch_size='auto', learning_rate='constant', learning_rate_init=1e-0, max_iter=3000, random_state=None)

# # Train the model
# model.fit(X, y)

# # Print loss
# loss = model.loss_
# print("Final loss:", loss)
