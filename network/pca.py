from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pathlib as pl

# define data
paths = ['datasets/normalized/A3-top10s.csv', 'datasets/preprocessed/A3-data.csv']

for file in paths:
    data = pd.read_csv(file, encoding='utf8', sep=';')
    # initialize PCA model
    pca = PCA(n_components=2)


    # fit and transform data
    data_transform = pca.fit_transform(data)


    # plot results
    plt.scatter(data_transform[:, 0], data_transform[:, 1], c=data.iloc[:,4])

    if 'A3-top10s' in file:
        plt.title('A3-top10s')
    else:
        plt.title('A3-data')
    plt.show()

    print(pca.explained_variance_ratio_)
