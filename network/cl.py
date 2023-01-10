from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import pandas as pd

paths = ['datasets/normalized/A3-top10s.csv', 'datasets/preprocessed/A3-data.csv']

for file in paths:
    data = pd.read_csv(file, encoding='utf8', sep=';')  
    
    # method='complete' should be complete linkage
    Z = linkage(data, method='complete')

    plt.figure(figsize=(25, 10))
    if 'A3-top10s' in file:
        plt.title('A3-top10s')
        dendrogram(Z, color_threshold=1.95)
        
    else:
        plt.title('A3-data')
        dendrogram(Z, color_threshold=16)
    plt.show()
