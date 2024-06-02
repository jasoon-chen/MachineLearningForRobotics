import numpy as np
import matplotlib.pyplot as plt

class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.centroids = None
    
    @staticmethod
    def euclidean_distance(data_point, centroids):
        # Compute Euclidian distance from one data point and all the centroids
        return np.sqrt(np.sum((centroids-data_point)**2, axis=1))
    
    def fit(self, X, max_iterations=300):
        # Generate random centroids within the min and max of the axis boundaries
        # self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))
        self.centroids = np.array([[3,5],[5,4],[7,6]])
        for _ in range(max_iterations): 
            y = []

            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                # print(distances)

                # Find the index of the point that has the closest distance to a centroid
                cluster_num = np.argmin(distances)
                # print(cluster_num)
                y.append(cluster_num)
            # print(y)
            y = np.array(y)

            # Remap all of the centroids
            cluster_indicies = []

            for i in range(self.k):
                cluster_indicies.append(np.argwhere(y == i))
            # print(cluster_indicies)
            
            cluster_centers = []
            for i, indicies in enumerate(cluster_indicies):
                # If the clusters are empty, set new centroid as old centroid
                if len(indicies) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    # Take the average position 
                    # print("------------------------")
                    # print(X[indicies])
                    # print(np.mean(X[indicies], axis=0))
                    # print(np.mean(X[indicies], axis=0)[0])
                    cluster_centers.append(np.mean(X[indicies], axis=0)[0])

            # If the difference between old and new centroids is less than 0.0001, we are done
            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                # Update the Centroids
                self.centroids = np.array(cluster_centers)

        return y

# X = np.random.randint(0, 100, (100,2))
X = np.array([[2, 3],
                   [3, 3],
                   [6, 8],
                   [8, 8],
                   [5, 6],
                   [7, 7]])

kmeans = KMeansClustering(k=3)
labels = kmeans.fit(X)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(kmeans.centroids[:,0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker="*", s=200)
plt.show()