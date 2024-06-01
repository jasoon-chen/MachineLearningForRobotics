import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# load the data set
data = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/test_brakes.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
data = data.drop(['payload'], axis=1)

y = (data.iloc[:,:-1].values).flatten()
x = (data.iloc[:,1].values).flatten()

# Computing Lasso
modelLasso1 = make_pipeline(
    PolynomialFeatures(8),
    Lasso(alpha=1e-15, max_iter=1e5)
)

modelLasso2 = make_pipeline(
    PolynomialFeatures(8),
    Lasso(alpha=0.01, max_iter=1e5)
)

modelLasso3 = make_pipeline(
    PolynomialFeatures(8),
    Lasso(alpha=10, max_iter=1e5)
)

# Pipeline: Compute Ridge
modelRidge1 = make_pipeline(
    PolynomialFeatures(8),
    Ridge(alpha=1e-15)
)

modelRidge2 = make_pipeline(
    PolynomialFeatures(8),
    Ridge(alpha=0.01)
)

modelRidge3 = make_pipeline(
    PolynomialFeatures(8),
    Ridge(alpha=10)
)

# training
modelLasso1.fit(x.reshape(-1, 1), y)
modelLasso2.fit(x.reshape(-1, 1), y)
modelLasso3.fit(x.reshape(-1, 1), y)
modelRidge1.fit(x.reshape(-1, 1), y)
modelRidge2.fit(x.reshape(-1, 1), y)
modelRidge3.fit(x.reshape(-1, 1), y)


plt.figure(figsize=(18, 14))
# plot Lasso
plt.subplot(231)
plt.scatter(x, y, color='y', marker='.')
plt.plot(x, modelLasso1.predict(x.reshape(-1, 1)), 'r-')
plt.title('Lasso Regression (alpha=1e-15)')

# plot Lasso
plt.subplot(232)
plt.scatter(x, y, color='y', marker='.')
plt.plot(x, modelLasso2.predict(x.reshape(-1, 1)), 'r-')
plt.title('Lasso Regression (alpha=0.01)')


# plot Lasso
plt.subplot(233)
plt.scatter(x, y, color='y', marker='.')
plt.plot(x, modelLasso3.predict(x.reshape(-1, 1)), 'r-')
plt.title('Lasso Regression (alpha=10)')


# plot Ridge
plt.subplot(234)
plt.scatter(x, y, color='y', marker='.')
plt.plot(x, modelRidge1.predict(x.reshape(-1, 1)), 'r-')
plt.title('Ridge Regression (alpha=1e-15)')

# plot Ridge
plt.subplot(235)
plt.scatter(x, y, color='y', marker='.')
plt.plot(x, modelRidge2.predict(x.reshape(-1, 1)), 'r-')
plt.title('Ridge Regression (alpha=0.01)')


# plot Ridge
plt.subplot(236)
plt.scatter(x, y, color='y', marker='.')
plt.plot(x, modelRidge3.predict(x.reshape(-1, 1)), 'r-')
plt.title('Ridge Regression (alpha=10)')

plt.show()