from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

paths = ['datasets/normalized/A3-top10s.csv', 'datasets/preprocessed/A3-data.csv']

for file in paths:
    data = pd.read_csv(file, encoding='utf8', sep=';')

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)

    labels = kmeans.predict(data)

   
    if 'A3-top10s' in file:
        data.plot.scatter(x="bpm", y="nrgy", c=labels)
        plt.title('A3-top10s')
    else:
        data.plot.scatter(x="x", y="y", c=labels)
        plt.title('A3-data')
    plt.show()
    