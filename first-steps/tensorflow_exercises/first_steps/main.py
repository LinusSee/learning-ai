import math

#from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


# Load data into panda dataframe
california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

# Shuffle data and rescale median_house_value (units of thousands)
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))
california_housing_dataframe["median_house_value"] /= 1000.0
print(california_housing_dataframe.iloc[:2])

# Prints out a statistic about the data (e.g. count, mean, min, max...)
print(california_housing_dataframe.describe())


my_feature = california_housing_dataframe[["total_rooms"]]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]

targets = california_housing_dataframe["median_house_value"]

my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000_000_1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

linear_regressor = tf.estimator.LinearRegressor(feature_columns=feature_columns, optimizer=my_optimizer)


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
	# Convert to dict of numpy arrays
	features = { key: np.array(value) for key, value in dict(features).items() }

	# Construct dataset and batching/repeating
	ds = Dataset.from_tensor_slices((features, targets)) # !! 2GB limit!!
	ds = ds.batch(batch_size).repeat(num_epochs)

	if shuffle:
		ds = ds.shuffle(buffer_size=10_000)

	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels


_ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature, targets), steps=100)

prediction_input_fn = lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=prediction_input_fn)
predictions = np.array([ item["predictions"][0] for item in predictions ])

mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)


min_house_value = california_housing_dataframe["median_house_value"].min()
max_house_value = california_housing_dataframe["median_house_value"].max()
min_max_difference =  max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min and Max: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

calibration_data = pd.DataFrame()
calibration_data["predictions"] = pd.Series(predictions)
calibration_data["targets"] = pd.Series(targets)
print(calibration_data.describe())

sample = california_housing_dataframe.sample(n=300)
x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()

weight = linear_regressor.get_variable_value("linear/linear_model/total_rooms/weights")[0]
bias = linear_regressor.get_variable_value("linear/linear_model/bias_weights")

y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

plt.plot([x_0, x_1], [y_0, y_1], c="r")
plt.xlabel("median_house_value")
plt.xlabel("total_rooms")
plt.scatter(sample["total_rooms"], sample["median_house_value"])
plt.show()
