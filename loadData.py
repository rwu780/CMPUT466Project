from __future__ import division
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

### Main Load Function

class DataLoader:

	def __init__(self, filename):
		self.dict = {}
		self.dataset = None
		self.loadData(filename)


	def loadData(self, filename):

		self.dataset = pd.read_csv(filename, delimiter = ';')
		self.dataset = self.processingData(self.dataset)

	def splitInputOutputData(self):
		'''
		Split data into inputs and outputs
		'''
		numOfFeatures = len(self.dataset.columns.values) - 1
		features = self.dataset.values[:, :numOfFeatures]
		target = self.dataset.values[:, numOfFeatures]

		return features, target


	def processingData(self, dataset):
		'''
		Convert catagorial value into numeric

		'''
		le = preprocessing.LabelEncoder()
		
		dup = dataset

		for column in dataset.columns.values:

			# Not numeric
			if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
				dataset[column] = le.fit_transform(dup[column])
			
		return dataset

	def splitTrainAndTestData(self):
		'''
		Split dataset into train and test data set
		'''

		numOfFeatures = len(self.dataset.columns.values) - 1
		features = self.dataset.values[:, :numOfFeatures]
		target = self.dataset.values[:, numOfFeatures]

		Xtrain, Xtest, ytrain, ytest = train_test_split(features, target, test_size = 0.33)

		return (Xtrain, ytrain), (Xtest, ytest)

