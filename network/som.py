from sklearn_som.som import SOM
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

# Load dataset
dataset = pd.read_csv('datasets/preprocessed/A3-data.csv', encoding='utf8', sep=';')
# data is everythin except the label column named class
data = dataset.drop("class", axis=1).values
# select label column named class
label = dataset["class"].values
print(data)
print(label)
# Extract just two features (just for ease of visualization)
data = data[:, :2]

som = SOM(5, 5, 2, sigma=1.0, lr=0.5)

# Fit it to the data
som.fit(data)

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