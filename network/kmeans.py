from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X = [[5,3], [10,15], [15,12], [24,10], [30,45], [85,70], [71,80], [60,78], [55,52], [80,91]]

# use k-means to classify the patterns in k = 2, 3, …, K classes
# K = 5, because?
K = 5
for k in range(2,K) :
        
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)

    labels = kmeans.predict(X)

    colors = ['r', 'g']
    for i in range(len(X)):
        plt.scatter(X[i][0], X[i][1], c=colors[labels[i]])
    plt.show()
