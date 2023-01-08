from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# define data
data = pd.read_csv(r'/home/m/NEC/A3/unsupervised-learning/datasets/raw/A3-data.txt')
# initialize PCA model
pca = PCA(n_components=2)


# fit and transform data
data_transform = pca.fit_transform(data)


# plot results
plt.scatter(data_transform[:, 0], data_transform[:, 1], c=data.iloc[:,4])
plt.show()

print(pca.explained_variance_ratio_)
