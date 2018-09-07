import pandas as pd

import classes.KnnNetwork as KnnNetwork


# Import iris dataset
data = pd.read_csv("assets/data/iris.csv")

kn = KnnNetwork.KnnNetwork()

#print("Single entry", data.iloc[0])
#print("Several entries", data.iloc[:5])
#print("Several keys", data.iloc[:5, :-1])
#print("Several names", data.iloc[:5, -1:])
#print(data.head())


def processData(data):
	keys = [ x[-1] for x in data.values.tolist() ]
	entries = [ x[:-1] for x in data.values.tolist() ]
	return list(zip(keys, entries))

trainingSet = processData(data.iloc[:105])
#validationSet = processData(data.iloc[105:135])
#testSet = processData(data.iloc[135:])

kn.train(trainingSet)

print(trainingSet[0])
correct = 0
for key, values in trainingSet:
	#print("Guess", kn.guess(values, 1))
	#print("Actual", key)
	#break
	if kn.guess(values, 1) == key:
		correct += 1

print(correct, "/", len(trainingSet))
