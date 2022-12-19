from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# define data
X = [[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0],
     [2.3, 2.7], [2.0, 1.6], [1.0, 1.1], [1.5, 1.6], [1.1, 0.9]]

# initialize PCA model
pca = PCA(n_components=2)


# fit and transform data
X_transform = pca.fit_transform(X)


# plot results
plt.scatter(X_transform[:, 0], X_transform[:, 1], c='red', edgecolors='none', s=30)
plt.show()

print(pca.explained_variance_ratio_)
