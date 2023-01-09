from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('datasets/preprocessed/A3-data.csv', encoding='utf8', sep=';')

# method='complete' should be complete linkage
Z = linkage(data, method='complete')

plt.figure(figsize=(25, 10))
dendrogram(Z)
plt.show()
