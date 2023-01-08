from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'/home/m/NEC/A3/unsupervised-learning/datasets/raw/A3-data.txt')

# method='complete' should be complete linkage
Z = linkage(data, method='complete')

plt.figure(figsize=(25, 10))
dendrogram(Z)
plt.show()
