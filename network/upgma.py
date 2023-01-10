from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

# define data
paths = ['datasets/normalized/A3-top10s.csv', 'datasets/preprocessed/A3-data.csv']

for file in paths:
    data = pd.read_csv(file, encoding='utf8', sep=';')

    # method='average' should be UPGMA
    Z = linkage(data, method='average')

    plt.figure(figsize=(25, 10))
   

    if 'A3-top10s' in file:
        dendrogram(Z, color_threshold=1.3)
        plt.title('A3-top10s')
    else:
        dendrogram(Z, color_threshold=9.5)
        plt.title('A3-data')
    plt.show()

