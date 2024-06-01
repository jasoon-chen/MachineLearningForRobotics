import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#our data set
data = pd.read_csv('/home/user/catkin_ws/src/results/test.csv')
xx = (data.iloc[:,:-1].values).flatten()
yy = (data.iloc[:,1].values).flatten()

#first we plot the data set
plt.figure(figsize=(10, 10))
plt.scatter(xx,yy, label = 'ground true')
plt.xlabel("X", fontsize=16)
plt.ylabel("Y", fontsize=16)
plt.title("Robot position", fontsize=18)

#now we compute the polynomial. We use numpy package
order3=np.polyfit(xx,yy,3)
order1=np.polyfit(xx,yy,1)
xp=np.linspace (0,3,100)

print('Single value prediction:', np.polyval(order3, 1.5))
print('Single value prediction:', np.polyval(order1, 1.5))

#we plot all togheter
plt.plot(xp,np.polyval(order3,xp),'--g', label = 'order 3')
plt.plot(xp,np.polyval(order1,xp),'--r', label = 'order 1')
plt.legend()
plt.show()