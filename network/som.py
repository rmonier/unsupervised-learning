from sklearn_som import SOM

model = SOM(5, 5, 2, sigma=1.0, learning_rate=0.5)
X = [[5,3], [10,15], [15,12], [24,10], [30,45], [85,70], [71,80], [60,78], [55,52], [80,91]]
model.fit(X)

model.visualize()
