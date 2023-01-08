from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv(r'/home/m/NEC/A3/unsupervised-learning/datasets/raw/A3-data.txt')

kmeans = KMeans(n_clusters=6)
kmeans.fit(data)

labels = kmeans.predict(data)

data.plot.scatter(x="x", y="y", c=labels)
plt.show()