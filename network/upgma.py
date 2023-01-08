from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

# define data
data = pd.read_csv(r'/home/m/NEC/A3/unsupervised-learning/datasets/raw/A3-data.txt')

# method='average' should be UPGMA
Z = linkage(data, method='average')

plt.figure(figsize=(25, 10))
dendrogram(Z)
plt.show()
