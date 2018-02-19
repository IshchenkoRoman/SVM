import os
import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm

class SVM():

	def __init__(self, path_data1):
		
		try:
			self._df1 = loadmat(path_data1)
		except IOError:
			print("File doesn't exist")
			raise

		self.X = self._df1['X']
		self.y = self._df1['y']

	def plotData(self):

		y1 = self.y.flatten()
		pos = y1 == 1
		neg = y1 == 0

		print(self.X.shape)
		plt.plot(self.X[:,0][pos], self.X[:,1][pos], "k+", markersize=7)
		plt.plot(self.X[:,0][neg], self.X[:,1][neg], "yo", markersize=7)

		plt.xlim(0, 5)
		plt.ylim(0, 5)

		plt.show()

	def svmTrain(self, X, y, C_=1, tol=0.001, max_passes=20):

		y1 = y.ravel()
		clf = svm.SVC(C=C_, tol=tol, max_iter=max_passes, kernel='linear')
		clf.fit(X, y1)
		return (clf)

	def plotVisualizeBoundary(self, X, y, model):

		data = model.decision_function(X)
		
		plt.scatter(X[:,0], X[:,1], s=50, c=data, cmap='seismic')

		x1plot = np.linspace(min(X[:,0])-0.25, max(X[:,0])+0.25, 100)
		x2plot = np.linspace(min(X[:,1])-0.25, max(X[:,1])+0.25, 100)

		X1, X2 = np.meshgrid(x1plot, x2plot)

		Z = model.predict(np.c_[X1.ravel(), X2.ravel()])
		Z = Z.reshape(X1.shape)

		plt.contourf(X1, X2, Z, cmap=plt.cm.Paired, alpha=0.2)		

		sv = model.support_vectors_
		plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidth='1')

		# y1 = self.y.flatten()
		# pos = y1 == 1
		# neg = y1 == 0

		# print(self.X.shape)
		# plt.plot(self.X[:,0][pos], self.X[:,1][pos], "k+", markersize=7)
		# plt.plot(self.X[:,0][neg], self.X[:,1][neg], "yo", markersize=7)

		# plt.xlim(0, 5)
		# plt.ylim(0, 5)

		plt.show()




def main():
	
	"""
	Part one- load and plot data
	"""
	curr_path = os.getcwd()
	path_data1 = curr_path + '/data/ex6data1.mat'

	svm = SVM(path_data1)
	# svm.plotData()

	"""
	Part two- Training Linear SVM
	"""

	C = 1
	model = svm.svmTrain(svm.X, svm.y, C)
	svm.plotVisualizeBoundary(svm.X, svm.y, model)


if __name__ == "__main__":
	main()