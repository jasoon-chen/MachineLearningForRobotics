import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import os

laser_range = 1.0
t = np.linspace(-laser_range, laser_range, 10)
nr_of_points = 1000
offset = 0.5

angle = 1.3962

laser1_line = np.tan(angle) #-offset
laser3_line = -np.tan(angle)#-offset
laser1 = []
laser2 = []
laser3 = []
laser_dead_zone = []

y_laser1 = np.ones(nr_of_points, dtype='uint8')*0
y_laser2 = np.ones(nr_of_points, dtype='uint8')*1
y_laser3 = np.ones(nr_of_points, dtype='uint8')*0 #2
y_laser_dead_zone = np.ones(5*nr_of_points, dtype='uint8')*0

## laser1

while (len(laser1)<nr_of_points):
    x = np.random.uniform(-laser_range, laser_range)
    y = np.random.uniform(-laser_range, laser_range)
    xy_rand = [x,y]
    point_vs_line =xy_rand[0] * laser1_line - offset

 

    if point_vs_line > xy_rand[1] and xy_rand[1] > 0:
        laser1.append(xy_rand)
    
        
##brain laser2        
        
while (len(laser2)<nr_of_points):
    x = np.random.uniform(-laser_range, laser_range)
    y = np.random.uniform(-laser_range, laser_range)
    xy_rand = [x,y]

    point_vs_line1 =xy_rand[0] * laser1_line - offset
    point_vs_line3 =xy_rand[0] * laser3_line - offset

  

    if point_vs_line1 < xy_rand[1] and xy_rand[1] > 0 and point_vs_line3 < xy_rand[1]:
        laser2.append(xy_rand)        

## laser3

while (len(laser3)<nr_of_points):
    x = np.random.uniform(-laser_range, laser_range)
    y = np.random.uniform(-laser_range, laser_range)
    xy_rand = [x,y]
    point_vs_line =xy_rand[0] * laser3_line - offset

  

    if point_vs_line > xy_rand[1] and xy_rand[1] > 0:
        laser3.append(xy_rand)
        
## dead zone of laser

while (len(laser_dead_zone)<5*nr_of_points):
    x = np.random.uniform(-laser_range, laser_range)
    y = np.random.uniform(laser_range*0.98, 2*laser_range)
    xy_rand = [x,y]
    #point_vs_line =xy_rand[0] * laser3_line  

    if laser_range < xy_rand[1]:
        laser_dead_zone.append(xy_rand)

        
        
laser1 = np.asarray(laser1)
laser2 = np.asarray(laser2)
laser3 = np.asarray(laser3)
laser_dead_zone = np.asarray(laser_dead_zone)

plt.figure(figsize=(10,10))
plt.scatter(laser1[:,0], laser1[:,1])
plt.scatter(laser2[:,0], laser2[:,1])
plt.scatter(laser3[:,0], laser3[:,1])
plt.scatter(laser_dead_zone[:,0], laser_dead_zone[:,1])

plt.plot(t, t*np.tan(angle)-offset, t, -t*np.tan(angle)-offset)
plt.ylim(bottom = 0, top = 2.1)
plt.xlim(left = -1, right = 1)

# plt.show()

X = np.concatenate((laser1, laser2, laser3, laser_dead_zone))
y = np.concatenate((y_laser1, y_laser2, y_laser3, y_laser_dead_zone))

N = 200 # number of points per class
D = 2 # dimensionality
K = 2 # number of classes
h = 1000 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
step_size = 1e-0


num_examples = X.shape[0]
ii = []
error = []

#here in for-loop we deploy the back propagation algorithm (gradient descent)

for i in range(1000):
  
  # evaluate class scores, [N x K]
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2
  
  # compute the Softmax class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss (cross-entropy loss)
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    #reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss # + reg_loss
    if i % 100 == 0:
        print ("i:", i, "loss: ", loss)
        ii.append(i)
        error.append(loss)
  
  # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
  
  # apply the backpropate algorithm(BP) the gradient 
    
  # first apply into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
  # next BP into hidden layer
    dhidden = np.dot(dscores, W2.T)
  # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
  # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
  
  # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

h = 0.03
x_min, x_max = X[:, 0].min() - 0, X[:, 0].max() + 0
y_min, y_max = X[:, 1].min() - 0, X[:, 1].max() + 0
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W) + b), W2) + b2
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8,8))
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

def save_weights():
    try:
        os.mkdir('/home/user/catkin_ws/src/weights_proj')

    except:
        print("Folder already exist")
        pass
    np.save('/home/user/catkin_ws/src/weights_proj/w2.npy',W2)
    np.save('/home/user/catkin_ws/src/weights_proj/b2.npy',b2)
    np.save('/home/user/catkin_ws/src/weights_proj/w.npy',W)
    np.save('/home/user/catkin_ws/src/weights_proj/b.npy',b)
    


save_weights() 