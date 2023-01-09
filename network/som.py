from sklearn_som.som import SOM
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import argparse

# Get parameters
parser = argparse.ArgumentParser(
                    prog = 'Unsupervised Learning SOM',
                    description = 'SOM for Unsupervised Learning')

parser.add_argument('vertical', type=int, default=5,
                    help='The number of vertical nodes in the SOM grid.')
parser.add_argument('horizontal', type=int, default=5,
                    help='The number of horizontal nodes in the SOM grid.')
parser.add_argument('dataset', type=str, default='datasets/preprocessed/A3-data.csv',
                    help='The dataset to use')
parser.add_argument('lr', type=float, default=0.5,
                    help='The initial step size for updating the SOM weights.')
parser.add_argument('sigma', type=float, default=1.0,
                    help='Magnitude of change to each weight. Does not update over training (as does learning rate). Higher values mean more aggressive updates to weights.')
parser.add_argument('classname', type=str, default='class',
                    help='The class column name')
parser.add_argument('epochs', type=int, default=1,
                    help='The number of epochs to train the SOM for.')

args = parser.parse_args()

# Load dataset
dataset = pd.read_csv(args.dataset, encoding='utf8', sep=';')
# data is everythin except the label column named class
data = dataset.drop(args.classname, axis=1).values
# select label column named class
label = dataset[args.classname].values
# get number of features
number_of_features = data.shape[1]

som = SOM(args.vertical, args.horizontal, number_of_features, sigma=args.sigma, lr=args.lr)

# Fit it to the data
som.fit(data, epochs=args.epochs)

predictions = som.predict(data)

# Plot the results
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,7))
x = data[:,0]
y = data[:,1]
colors = ['red', 'green', 'blue']

ax[0].scatter(x, y, c=label, cmap=ListedColormap(colors))
ax[0].title.set_text('Actual Classes')
ax[1].scatter(x, y, c=predictions, cmap=ListedColormap(colors))
ax[1].title.set_text('SOM Predictions')

plt.show()