from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

# define data
data = pd.read_csv('datasets/preprocessed/A3-data.csv', encoding='utf8', sep=';')

# method='average' should be UPGMA
Z = linkage(data, method='average')

plt.figure(figsize=(25, 10))
dendrogram(Z)
plt.show()
