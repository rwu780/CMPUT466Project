import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

class Classifier:

	def __init__(self, parameters = {}):
		self.params = {}
		self.reset(parameters)

	def reset(self, parameters):
		self.weights = None
		self.resetparams(parameters)

	def resetparams(self, parameters):
		self.weights = None
		try:
			self.updateDictionary(self.params, parameters)
		except AttributeError:
			self.params = {}

	def updateDictionary(self, dict1, dict2):

		for k in dict1:
			if k in dict2:
				dict1[k] = dict2[k]


	def getparams(self):
		return self.params

	def learn(self, Xtrain, ytrain):
		"""Learns using the train data"""

	def predict(self, Xtest, ytest = None):

		probs = np.random.rand(Xtest.shape[0])
		ytest = np.ones(len(probs),)

		for i in range(len(ytest)):
			if probs[i] < 0.5:
				ytest[i] = 0

		return ytest

	def getAlg(self):
		return self.alg

class SVMClassifier(Classifier):

	def __init__(self, parameters = {}):
		self.params = {'regwgt':0.0}
		self.reset(parameters)
		self.alg = LinearSVC()

	def learn(self, Xtrain, ytrain):
		self.alg.fit(Xtrain, ytrain)

	def predict(self, Xtest):
		ytest = self.alg.predict(Xtest)

		return ytest

class LogisticRegressionClassifier(Classifier):

	def __init__(self, parameters = {}):
		self.params = {'regwgt': 0.0, 'regularizer': 'None'}
		self.reset(parameters)
		self.alg = LogisticRegression()

	def reset(self, parameters):

		self.resetparams(parameters)
		self.weights = None
		if self.params['regularizer'] is 'l1':
			self.alg = LogisticRegression(penalty = 'l1', solver= 'saga')
		elif self.params['regularizer'] is 'l2':
			self.alg = LogisticRegression(penalty = 'l2', solver = 'sag', max_iter=1000)
		elif self.params['regularizer'] is None:
			self.alg = LogisticRegression(penalty = 'l2', solver = 'sag', max_iter=1000, C = 10000)

	def learn(self, Xtrain, ytrain):
		self.alg.fit(Xtrain, ytrain)

	def predict(self, Xtest):
		ytest = self.alg.predict(Xtest)
		return ytest

class NeuralNetwork(Classifier):

	def __init__(self, parameters = {}):
		self.params = {'regwgt':0.0, 'nh':(300,1),
						'transfer': 'sigmoid',
						'stepsize': 0.01,
						'epochs': 200
							}
		self.reset(parameters)
		self.alg = MLPClassifier()

	def reset(self, parameters):
		self.resetparams(parameters)
		self.transfer = ''
		if self.params['transfer'] is 'sigmoid':
			self.transfer = 'sigmoid'
			self.alg = MLPClassifier(hidden_layer_sizes = self.params['nh'],
									 activation = 'logistic',
									 solver = 'sgd',
									 learning_rate = 'constant',
									 learning_rate_init = self.params['stepsize'],
									 max_iter = self.params['epochs'])
	def learn(self, Xtrain, ytrain):
		self.alg.fit(Xtrain, ytrain)

	def predict(self, Xtest):
		ytest = self.alg.predict(Xtest)
		return ytest












		