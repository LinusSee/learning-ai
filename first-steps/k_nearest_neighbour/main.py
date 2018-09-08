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
def run(data, k, printLabel):
	correct = 0
	for key, values in data:
		if kn.guess(values, k) == key:
			correct += 1

	print(printLabel, correct, "/", len(data))


run(trainingSet, 1, "Train1")
run(validationSet, 1, "Validate1")
run(trainingSet, 3, "Train3")
run(validationSet, 3, "Validate3")
run(trainingSet, 5, "Train5")
run(validationSet, 5, "Validate5")
run(trainingSet, 7, "Train7")
run(validationSet, 7, "Validate7")

#print(kn.guess(validationSet[0][1], 1))
#print(kn.guess(validationSet[0][1], 3))
#print(kn.guess(validationSet[0][1], 5))
#print(kn.guess(validationSet[0][1], 7))
test = [7.2, 3.6, 5.1, 2.5]
print(kn.guess(test, 1))
print(kn.guess(test, 3))
print(kn.guess(test, 5))
print(kn.guess(test, 7))

clazz = KNeighborsClassifier(n_neighbors=1)
clazz.fit(data.iloc[:105, 0:4], data.iloc[:105, -1])
print("SciKit", clazz.predict([test]))

clazz = KNeighborsClassifier(n_neighbors=3)
clazz.fit(data.iloc[:105, 0:4], data.iloc[:105, -1])
print("SciKit", clazz.predict([test]))

clazz = KNeighborsClassifier(n_neighbors=5)
clazz.fit(data.iloc[:105, 0:4], data.iloc[:105, -1])
print("SciKit", clazz.predict([test]))

clazz = KNeighborsClassifier(n_neighbors=7)
clazz.fit(data.iloc[:105, 0:4], data.iloc[:105, -1])
print("SciKit", clazz.predict([test]))
