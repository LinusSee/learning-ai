import math

from IPython import display
from matplotlib import cm
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

def preprocess_features(california_housing_dataframe):
	selected_features = california_housing_dataframe[
		[
			"latitude",
			"longitude",
			"housing_median_age",
			"total_bedrooms",
			"population",
			"households",
			"median_income"
		]
	]
	processed_features = selected_features.copy()
	processed_features["rooms_per_person"] = (california_housing_dataframe["total_rooms"] / california_housing_dataframe["population"])
	return processed_features

def preprocess_targets(california_housing_dataframe):
	output_targets = pd.DataFrame()
	output_targets["median_house_value"] = (california_housing_dataframe["median_house_value"] / 1000.0)
	return output_targets

training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))

validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["median_house_value"]
print(correlation_dataframe.corr())


def construct_feature_columns(input_features):
	return set([ tf.feature_column.numeric_column(feature) for feature in input_features ])

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
	features = { key: np.array(value) for key, value in dict(features).items() }

	ds = Dataset.from_tensor_slices((features, targets))
	ds = ds.batch(batch_size).repeat(num_epochs)

	if shuffle:
		ds = ds.shuffle(10000)

	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

def train_model(learning_rate, steps, batch_size, training_examples, training_targets, validation_examples, validation_targets):
	periods = 10
	steps_per_period = steps / periods

	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
	linear_regressor = tf.estimator.LinearRegressor(feature_columns=construct_feature_columns(training_examples), optimizer=my_optimizer)

	training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"], batch_size=batch_size)
	predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"], num_epochs=1, shuffle=False)
	predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["median_house_value"], num_epochs=1, shuffle=False)


	print("Training model...")
	print("RMSE (on training data):")
	training_rmse = []
	validation_rmse = []
	for period in range(0, periods):
		linear_regressor.train(input_fn=training_input_fn, steps=steps_per_period)

		training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
		training_predictions = [ item["predictions"][0] for item in training_predictions ]

		validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
		validation_predictions = [ item["predictions"] for item in validation_predictions ]

		training_root_mse = math.sqrt(metrics.mean_squared_error(training_predictions, training_targets))
		validation_root_mse = math.sqrt(metrics.mean_squared_error(validation_predictions, validation_targets))

		print("	period %02d : %0.2f" % (period, training_root_mse))
		training_rmse.append(training_root_mse)
		validation_rmse.append(validation_root_mse)

	print("Model training finished.")

	plt.ylabel("RMSE")
	plt.xlabel("Periods")
	plt.title("Root Mean Squared Error vs. Periods")
	plt.tight_layout()
	plt.plot(training_rmse, label="training")
	plt.plot(validation_rmse, label="validation")
	plt.legend()
	plt.show()

	return linear_regressor


minimal_features = [
	"median_income",
	"rooms_per_person"
	#"latitude"
]

assert minimal_features, "Select at least one feature"

minimal_training_examples = training_examples[minimal_features]
minimal_validation_examples = validation_examples[minimal_features]

linear_regressor = train_model(
	learning_rate=0.1,
	steps=500,
	batch_size=5,
	training_examples=minimal_training_examples,
	training_targets=training_targets,
	validation_examples=minimal_validation_examples,
	validation_targets=validation_targets
)

# Here you could probably improve the performance further by binning the latitude
# Reason for this is, that at some latitudes (San Francisco and Los Angeles) the prices
# are higher. The google tutorial contains this process as well
# Another option I thought of would be creating a matrix of the latitude and longitude
# and then flattening it into a vector, which might yield even better results than binning

california_housing_test_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_test.csv", sep=",")
test_examples = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)

predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets, num_epochs=1, shuffle=False)

test_predictions = linear_regressor.predict(predict_test_input_fn)
test_predictions = np.array([ item["predictions"][0] for item in test_predictions ])
# Above: print an item to see what is in the other positions

root_mean_squared_error = math.sqrt(metrics.mean_squared_error(test_predictions, test_targets))

print("Final RSME (on test data): %0.2f" % root_mean_squared_error)
