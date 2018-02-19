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

	def __init__(self, path_data1, path_data2):
		
		try:
			self._df1 = loadmat(path_data1)
		except IOError:
			print("File doesn't exist")
			raise

		try:
			self._df2 = loadmat(path_data2)
		except IOError:
			print("File doent't exist")
			raise

		self.X = self._df1['X']
		self.y = self._df1['y']

		self.X1 = self._df2['X']
		self.y1 = self._df2['y']

	def plotData(self, X, y):

		y1 = y.flatten()
		pos = y1 == 1
		neg = y1 == 0

		plt.plot(X[:,0][pos], X[:,1][pos], "k+", markersize=7)
		plt.plot(X[:,0][neg], X[:,1][neg], "yo", markersize=7)

		plt.show()

	def svmTrain(self, X, y, C=1, tol=0.001, max_passes=20, type_kernel='linear', sigma=0.01):

		y1 = y.ravel()
		if (type_kernel == "precomputed"):
			clf = svm.SVC(C=C, tol=tol, max_iter=max_passes, kernel=type_kernel)
			model = clf.fit(self.gaussianKernelGramMatrix(X, X, sigma), y1)
		else:
			clf = svm.SVC(C=C, tol=tol, max_iter=max_passes, kernel=type_kernel)
			model = clf.fit(X, y1)
		return (model)

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

		plt.show()

	def gaussianKernel(self, x1, x2, sigma):

		res = x1 - x2
		return (np.exp(-(np.dot(res, res.T)) / (2 * (sigma ** 2))))

	def gaussianKernelGramMatrix(self, X1, X2, sigma=0.1):

		"""Pre calculated Gram Matrix K"""

		gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
		for i, x1 in enumerate(X1):
			x1 = x1.flatten()

			for j, x2 in enumerate(X2):
				x2 = x2.flatten()
				gram_matrix[i, j] = np.exp(- np.sum( np.power((x1 - x2), 2) ) / float(2 * (sigma ** 2)))

		return (gram_matrix)

	def plotVisualizeBoundaryMyGaussian(self, X, y, model):

		y1 = y.flatten()
		pos = y1 == 1
		neg = y1 == 0

		plt.plot(X[:,0][pos], X[:,1][pos], "k+", markersize=7)
		plt.plot(X[:,0][neg], X[:,1][neg], "yo", markersize=7)

		x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
		x2 = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
		X1, X2 = np.meshgrid(x1, x2)
		vals = np.zeros(X1.shape)
		for i in range(X1.shape[1]):
			this_x = np.c_[X1[:, i], X2[:, i]]
			vals[:, i] = model.predict(self.gaussianKernelGramMatrix(this_x, X))

		plt.contour(X1, X2, vals, colors="blue", levels=[0,0])
		plt.show()


def main():
	
	"""
	Part one- load and plot data
	"""
	curr_path = os.getcwd()
	path_data1 = curr_path + "/data/ex6data1.mat"
	path_data2 = curr_path + "/data/ex6data2.mat"

	svm = SVM(path_data1, path_data2)
	# svm.plotData(svm.X, svm.y)

	"""
	Part two- Training Linear SVM
	"""

	# C = 1
	# model = svm.svmTrain(svm.X, svm.y, C)
	# svm.plotVisualizeBoundary(svm.X, svm.y, model)
	# C_100 = 100
	# model_100 = svm.svmTrain(svm.X, svm.y, C_100)
	# svm.plotVisualizeBoundary(svm.X, svm.y, model_100)

	"""
	Part 3: Implement Gaussian Kernel
	"""

	# x1 = np.array([1,2,1])
	# x2 = np.array([0,4,-1])
	# sigma = 2
	# sim = svm.gaussianKernel(x1, x2, sigma)
	# print(sim)

	"""
	Part 4: load (done above) and visualise data
	"""

	# svm.plotData(svm.X1, svm.y1)

	"""
	Part 5: Training SVM with RBF Kernel (Dataset 2)
	"""

	C = 1
	sigma = 0.1
	model = svm.svmTrain(svm.X1, svm.y1, C, tol=0.001, max_passes=-1, type_kernel="precomputed", sigma=sigma)
	svm.plotVisualizeBoundaryMyGaussian(svm.X1, svm.y1, model)


if __name__ == "__main__":
	main()