from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd


model = TSNE(n_components=2)

data = pd.read_csv(r'/home/m/NEC/A3/unsupervised-learning/datasets/raw/A3-data.txt')
tsne_results = model.fit_transform(data)

plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data.iloc[:,4])
plt.show()


# colors = ['red', 'blue', 'green']
# for i, color in enumerate(colors):
#     x = tsne_results[:, 0][labels == i]
#     y = tsne_results[:, 1][labels == i]
#     plt.scatter(x, y, c=color)
# plt.show()
