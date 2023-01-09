from sklearn_som import SOM
import pandas as pd

data = pd.read_csv('datasets/raw/A3-data.txt')
model = SOM(5, 5, 2, sigma=1.0, learning_rate=0.5)
model.fit(data)

model.visualize()
