from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# define data
data = pd.read_csv('datasets/preprocessed/A3-data.csv', encoding='utf8', sep=';')
# initialize PCA model
pca = PCA(n_components=2)


# fit and transform data
data_transform = pca.fit_transform(data)


# plot results
plt.scatter(data_transform[:, 0], data_transform[:, 1], c=data.iloc[:,4])
plt.show()

print(pca.explained_variance_ratio_)
