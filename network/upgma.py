from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

X = [[5,3], [10,15], [15,12], [24,10], [30,45], [85,70], [71,80], [60,78], [55,52], [80,91]]

Z = linkage(X, method='average')

plt.figure(figsize=(25, 10))
dendrogram(Z)
plt.show()