import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

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
validationSet = processData(data.iloc[105:135])
testSet = processData(data.iloc[135:])

kn.train(trainingSet)

correctTraining = 0
def runOwn(data, k, printLabel):
	correct = 0
	for key, values in data:
		if kn.guess(values, k)[0] == key:
			correct += 1

	print(printLabel, correct, "/", len(data))

def runSciKit(valData, k, printLabel):
	correct = 0
	clazz = KNeighborsClassifier(n_neighbors=k)
	clazz.fit(data.iloc[:105, 0:4], data.iloc[:105, -1])
	for key, values in valData:
		if clazz.predict([values])[0] == key:
			correct += 1

	print(printLabel, correct, "/", len(valData))


test = [7.2, 3.6, 5.1, 2.5]

#clazz = KNeighborsClassifier(n_neighbors=5)
#clazz.fit(data.iloc[:105, 0:4], data.iloc[:105, -1])
#for key, values in validationSet:
#	print("Own", kn.guess(values, 5), " with actual ", key)
#	print("SciKit", clazz.predict([values]), clazz.kneighbors([values])[1], " with actual", key)
for x in range(1, 8, 2):
	runOwn(validationSet, x, "Own" + str(x))
	runSciKit(validationSet, x, "SciKit" + str(x))

#print(kn.guess(validationSet[0][1], 3))

