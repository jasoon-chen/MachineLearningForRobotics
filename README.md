# Machine Learning For Robotics
This is to help people self learn basic machine learning concepts through [The Construct](https://app.theconstruct.ai/courses/machine-learning-for-robotics-47/). Even though all of the tutorials and lessons are already on the website, this repository is meant to upload all of my solutions and also some hints while working through the project as some of them may not be clear. Feel free to create a github pull request or issue if you want to add onto this repository. The topics that are covered in this course are:
1. Linear/Logistic Regression and Regularization
2. Supervised Learning: Backpropagation Algorithm and Deploying NN in ROS
3. Unsupervised Learning: K-Means Algorithm
4. Principle Component Analaysis

# Status
I am currently in progress of working through this course.

# 2 - Linear and Logistic Regression and Regularization
Although the instructions are clear throughout this lesson, I feel like it does not do enough code explanation for those who are new to utilizing numpy or panda. The python syntax overall should be fine, although I do want to create some explanations on the code. In the section where it explains how to run a LinearRegression, I have added comments to help further understanding of the code. The entire code snippet will be attatched at the end.

I will start by explaining he code of cleaning and preparing the data.

```py
data.drop(['Unnamed: 0'], axis=1, inplace=True)
```

The first argument is used to define the labels or column that you want to remove. In this case, if you open the CSV file, you will notice how there is an empty column. Sometimes when a CSV file is being read onto a Pandas DataFrame, it creates an extra column and I'm pretty sure you would need to check to see if you need to remove the first column or instead of removing the first column adapting your code to start indexing from 1 instead of 0. 
The second argument `axis=1` is utilized to state whether to drop the labels from the row or column. If `axis=1`, then you are dropping the column. If `axis=0` then you are dropping the row.
The third argument `inplace=True` is utilized to determine whether to keep the original data or replace it. If `inplace=True` then it will replace the original data. If `inplace=False` then it will return a new data and keep the original dataFrame unchanged. Default value is False, or in other words it will return a new data.
[Link To pandas.DataFrame.drop Library](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html)

```py
yA = (data.iloc[:,0].values).flatten()
```

Now that we have removed the first column of our data, we can start indexing at 0 instead of 1. The `data.iloc()` function allows us to index based on the massive data that we have. In this case `[:,0]` syntax means that we are going to select all of the rows and only select the first column. This allows us to return distance column in the CSV file. Now since data is a Pandas DataFrame object, we need to convert this into a NumPy Array as these are two completely different libraries. In this case, we can use `.values` to do the conversion. The `.flatten()` helps convert the multi-dimensional array into one-dimensional array.

The rest of the code I believe is self-explanatory so I won't further explain the rest of the code. 
<details>
  <summary>Click Here For the Entire Code</summary>

  ```py
  import warnings
  warnings.filterwarnings('ignore')
  import matplotlib
  import matplotlib.pyplot as plt
  from matplotlib import cm
  from mpl_toolkits.mplot3d import Axes3D
  import numpy as np
  import pandas as pd
  
  # class definition
  
  class LinearRegression(object):
  
      def __init__(self):
  
          self._m = 0
          self._b = 0
      
      def m(self):
  
          return self._m
  
      def b(self):
  
          return self._b
  
      def fit(self, X, y): # IMPLEMENTATION ABOVE EQUATIONS TO COMPUTE: m, b  
  
          X = np.array(X)
          y = np.array(y)
          X_ = X.mean()
          y_ = y.mean()
          num = ((X - X_)*(y - y_)).sum()
          den = ((X - X_)**2).sum()
          self._m = num/den
          self._b = y_ - self._m*X_
  
      def predict(self, x):
  
          x = np.array(x)
          return self._m*x + self._b
  
  # Computation of MSE and regression (we use the same formulas as we defined earlier)
  
  def MSE(ax, x, y, model):
  
      error = y - model.predict(x)
      MSE = (error**2).sum()/error.size
      ax.plot([x, x], [error*0, error])
      return MSE
  
  def compute_regression(ax, x, y, model):
  
      error = y - model.predict(x)
      MSE = (error**2).sum()/error.size
      ax.scatter(x, y, label='distance')
      ax.plot([x, x], [y, model.predict(x)], ':')
      ax.plot(0, 0, ':', alpha=0.5, label='error')
      ax.plot([0, 100], model.predict([0, 100]), color='red', label='regression')
      ax.axis([0, 100, 0, 22])
      ax.legend()
  
  # model is a object of class
  model = LinearRegression()
  
  # load dataset
  data = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/test_brakes.csv')
  
  # remove Unmaned column
  data.drop(['Unnamed: 0'], axis=1, inplace=True)
  
  # our data set
  yA = (data.iloc[:,0].values).flatten()
  x = (data.iloc[:,1].values).flatten()
  
  plt.figure(figsize=(10, 8))
  axA = plt.axes(xlim=(0, 100), ylim=(0, 22), autoscale_on=False)
  model.fit(x, yA)
  compute_regression(axA, x, yA, model)
  plt.xlabel("% of max speed of axis 1", fontsize=16)
  plt.ylabel("stop distance [deg]", fontsize=16)
  plt.title("Linear regression", fontsize=18)
  
  plt.show()
  ```
</details>

Continuing onto the Gradient Descent section, I believe that there is a mistake with the code. When finding the MSE for the new m and b values, the function that is provided I believe is incorrect. This is the code that is provided: 
```py
def compute_error_for_line_given_points(b,m,X,Y):
    totalError = 0 	#sum of square error formula
    for i in range (0, len(X)):
        x = X
        y = Y
        totalError += (y[i]-(m*x[i] + b)) ** 2
        mse = (totalError)/totalError.size
        return mse
```
While this does compute the MSE for the first point, it does not compute the total MSE. To fix this, I believe that this is the correct solution which I have provided on the github profile. 
```py
def computeErrorForLineGivenPoints(self,b,m,X,Y):
        totalError = 0
        for i in range(0, len(X)):
            x = X
            y = Y
            functionResult = m*x[i] + b
            totalError += (y[i]-functionResult)**2
        return (totalError)/len(X)
```

I also understand that the math portion may be a bit unclear in the deriviation, so I have attatched the deriviation under here. 
<details>
  <summary>Click Here Gradient Descent Math Deriviation</summary>
  
  ![Gradient Descent Math Deriviation](https://github.com/jasoon-chen/MachineLearningForRobotics/assets/45437968/c40b87fd-0a2f-4f00-8bc6-88c5ca5d1e77)
</details>

For section 2.3, the code for plotting the mesh grid does not work. There appears to be some minor bugs and changes since it was last created, so I have attatched a Google Collab version of the code that should work as intended. There is also small comments that explain the code, however to be honest I still don't understand much of the code but I do hope to understand more of it as I go through the course. [Link to the Google Collab Notebook](https://colab.research.google.com/drive/10ITOg52E0z7LwvImpuonWSQcLmCbfYVK?usp=sharing)

For section 2.5, the explanation for Logistic Regression is not really well explained. So the following videos below will cover basically what is explained in the course.
1. [What is Logistic Regression?](https://www.youtube.com/watch?v=U1omz0B9FTw)
2. [Cost Function of Logistic Regression](https://www.youtube.com/watch?v=ar8mUO3d05w)
3. [Cross Entropy of Logistic Regression](https://youtube.com/watch?v=MztgenIfGgM)

For section 2.7/2.8, the explanation for L1 and L2 Regularization is not really well explained so I have attatched the following videos that explain both of them.
1. [L2 Regularization](https://www.youtube.com/watch?v=Q81RR3yKn30)
2. [L1 Regularization](https://www.youtube.com/watch?v=NGf0voTMlcs)
   
