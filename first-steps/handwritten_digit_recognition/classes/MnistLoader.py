import pickle
import gzip

import numpy as np

'''
	This code (unlike the network itself) is mostly taken from this page:
		http://neuralnetworksanddeeplearning.com/chap1.html
'''
class MnistLoader:
	def load_data(self):
			f = gzip.open("./assets/mnist/mnist.pkl.gz", "rb")
			training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
			f.close()
			return (training_data, validation_data, test_data)

	def load_data_wrapper(self):
		tr_d, va_d, te_d = self.load_data() # Training-, validation- and testdata

		# Change row to column vector
		training_inputs = [ np.reshape(x, (784, 1)) for x in tr_d[0] ]
		training_results = [ self.vectorized_result(y) for y in tr_d[1] ]
		training_data = zip(training_inputs, training_results)

		validation_inputs = [ np.reshape(x, (784, 1)) for x in va_d[0] ]
		validation_data = zip(validation_inputs, va_d[1])

		test_inputs = [ np.reshape(x, (784, 1)) for x in te_d[0] ]
		test_data = zip(test_inputs, te_d[1])

		return (training_data, validation_data, test_data)

	def vectorized_result(self, j):
		e = np.zeros((10, 1))
		e[j] = 1.0
		return e
