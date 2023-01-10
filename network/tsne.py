from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

paths = ['datasets/normalized/A3-top10s.csv', 'datasets/preprocessed/A3-data.csv']

for file in paths:
    data = pd.read_csv(file, encoding='utf8', sep=';')
    model = TSNE(n_components=2)

    tsne_results = model.fit_transform(data)

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=data.iloc[:,4])
    if 'A3-top10s' in file:
        plt.title('A3-top10s')
    else:
        plt.title('A3-data')
    plt.show()


    # colors = ['red', 'blue', 'green']
    # for i, color in enumerate(colors):
    #     x = tsne_results[:, 0][labels == i]
    #     y = tsne_results[:, 1][labels == i]
    #     plt.scatter(x, y, c=color)
    # plt.show()
