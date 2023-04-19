import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = np.array(
    [
        [-25,-46],
        [-22,-43],
        [-25,-49],
        [-30,-51],
        [-19,-43],
        [-15,-47],
        [-12,-38],
        [-8,-34],
        [-16,-49],
        [-3,-60],
        [-21,-47],
        [-23,-51],
        [-27,-48],
        [-21,-43],
        [-1,-48],
        [-10,-67],
        [-8,-63],
        [-22,-47],
        [-3,-38],
        [0,-33],
        [-2,-30],
        [-40,-70],
        [-19,-60],
        [-49,-40],
        [-40,-60],
        [-30,-65],
        [0,-66],
        [-50,-75],
        [-40,-50]
    ]
)
kmeans = KMeans(n_clusters=3, init='k-means++',n_init=10,max_iter=300)
predicao = kmeans.fit_predict(dataset)

plt.scatter(dataset[:,1], dataset[:,0], c=predicao)
plt.xlim(-75,-30)
plt.ylim(-50,10)
plt.grid()

print(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0])
plt.scatter(kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,0], s=70, c='red')
plt.show()