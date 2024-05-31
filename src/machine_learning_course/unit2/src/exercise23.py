import numpy as np
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D


data = pd.read_csv('/home/user/catkin_ws/src/machine_learning_course/dataset/test_brakes_multi.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)

x = data[['speed', 'payload']]
y = data[['distance']]

X1 = sm.add_constant(x)
est = sm.OLS(y,X1).fit()

xx1, xx2 = np.meshgrid(np.linspace(X1.speed.min(), X1.speed.max(), 100), 
                       np.linspace(X1.payload.min(), X1.payload.max(), 100))

xx1, xx2 = np.meshgrid(np.linspace(X1.speed.min(), 100, 100), 
                       np.linspace(X1.payload.min(), 50, 100))

Z = est.params[0] + est.params[1] * ( xx1 ** 1 ) + est.params[2] * ( xx2 ** 1 )
fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, azim=-115, elev=15)
surf = ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)
ax.scatter(X1.speed,X1.payload,y, color='black', alpha=1.0, facecolor='white')
ax.set_xlabel('speed', fontsize=16)
ax.set_ylabel('payload', fontsize=16)
ax.set_zlabel('distance', fontsize=16)
plt.title("Multi - regression",  fontsize=18)