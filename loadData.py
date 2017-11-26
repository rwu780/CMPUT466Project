from __future__ import division
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

### Main Load Function

class DataLoader:

	def __init__(self):
		self.dict = {}

	def loadData(self, filename):

		dataset = pd.read_csv(filename, delimiter = ';')
		dataset = self.processingData(dataset)
		trainData, testData = self.splitData(dataset)

		return trainData, testData

	def processingData(self, dataset):
		le = preprocessing.LabelEncoder()
		
		dup = dataset

		for column in dataset.columns.values:

			# Not numeric
			if dataset[column].dtype != np.int64 and dataset[column].dtype != np.float64:
				dataset[column] = le.fit_transform(dup[column])
			
		return dataset

	def splitData(self, dataset):
		numOfFeatures = len(dataset.columns.values) - 1
		features = dataset.values[:, :numOfFeatures]
		target = dataset.values[:, numOfFeatures]

		Xtrain, Xtest, ytrain, ytest = train_test_split(features, target, test_size = 0.33)

		return (Xtrain, ytrain), (Xtest, ytest)


