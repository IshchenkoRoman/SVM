import os
import os.path
import operator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import colors

from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm

class SVM():

	def __init__(self, path_data1, path_data2, path_data3):
		
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

		try:
			self._df3 = loadmat(path_data3)
		except IOError:
			print("File doent't exist")
			raise

		self.X = self._df1['X']
		self.y = self._df1['y']

		self.X1 = self._df2['X']
		self.y1 = self._df2['y']

		self.X2 = self._df3['X']
		self.y2 = self._df3['y']
		self.X2val = self._df3["Xval"]
		self.y2val = self._df3["yval"]


	def plotData(self, X, y):

		y1 = y.flatten()
		pos = y1 == 1
		neg = y1 == 0

		plt.plot(X[:,0][pos], X[:,1][pos], "k+", markersize=7)
		plt.plot(X[:,0][neg], X[:,1][neg], "yo", markersize=7)

		plt.show()

	def svmTrain(self, X, y, C=1, tol=0.001, max_passes=-1, type_kernel='linear', sigma=0.01):

		y1 = y.ravel()
		if (type_kernel == "precomputed"):
			clf = svm.SVC(C=C, tol=tol, max_iter=max_passes, kernel=type_kernel)
			model = clf.fit(self.gaussianKernelGramMatrix(X, X, sigma), y1)
		else:
			clf = svm.SVC(C=C, tol=tol, max_iter=max_passes, kernel=type_kernel)
			model = clf.fit(X, y1)
		return (model)

	def plotVisualizeBoundary(self, X, y, model, C):

		"""
		TODO:
		How add continuous discrete data to legend?
		"""

		ax = plt.subplot(1,1,1)
		ax.set_title("SVM (C={0}) Decision Confidence".format(C))
		xs = np.linspace(-1, 8)

		#Calculate decision boundary

		b = model.intercept_[0]
		w_0 = model.coef_[0, 0]
		w_1 = model.coef_[0, 1]
		a = -w_0 / w_1
		db_1 = a * xs - b / w_1

		#Store support vectors
		svs = model.support_vectors_

		#Calculate margins

		c = svs[0]
		margin_low = a * (xs - c[0]) + c[1] # Line of slope "a" passing through point "(c[0], c[1])"
		c = svs[-2]
		margin_high = a * (xs - c[0]) + c[1]

		# Plot data, margnin, decision boundaries, mark support vectors

		data = model.decision_function(X)
		# Here i plot continuous discrete data
		plt.scatter(X[:,0], X[:,1], s=50, c=data, cmap='seismic')
		dbl = plt.plot(xs, db_1, 'b-', lw=1, label="Decision boundary")
		ml = plt.plot(xs, margin_low, 'b--', lw=0.5, label="Margin")
		plt.plot(xs, margin_high, 'b--', lw=0.5)
		svl = plt.plot(svs.T[0], svs.T[1], marker='o', ls='none', ms='15', mfc='none', mec='b', mew=0.5, label="Support vectors")

		# y1 = y.flatten()
		# pos = y1 == 1
		# neg = y1 == 0

		# pl = plt.plot(X[:,0][pos], X[:,1][pos], "ro", markersize=7, label="Pos. examples")
		# nl = plt.plot(X[:,0][neg], X[:,1][neg], "yo", markersize=7, label="Neg. examples")

		# Plot areas higher and lower margin 

		x1plot = np.linspace(min(X[:,0])-1, max(X[:,0])+1, 100)
		x2plot = np.linspace(min(X[:,1])-1, max(X[:,1])+1, 100)
		X1, X2 = np.meshgrid(x1plot, x2plot)
		Z = model.predict(np.c_[X1.ravel(), X2.ravel()])
		Z = Z.reshape(X1.shape)
		plt.contourf(X1, X2, Z, cmap=plt.cm.Paired, alpha=0.25)		

		handlers, labels = ax.get_legend_handles_labels()
		hl = sorted(zip(handlers, labels), key=operator.itemgetter(1))
		h,l = zip(*hl)
		plt.legend(h, l, loc=3, framealpha=0.3)

		plt.xlim(0, 4.5)
		plt.ylim(1.5, 5)

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

		ax = plt.subplot(1,1,1)
		ax.set_title("SVM (Gaussian Kernel) Decision Boundary")

		# Little cheating for nice picture
		# svc = svm.SVC(C=100, kernel="rbf", gamma=6)
		svc = svm.SVC(C=100, gamma=10, probability=True)
		svc.fit(X, y.ravel())
		data = svc.predict_proba(X)[:,0]
		x_min, x_max = X[:,0].min() - 0.25, X[:,0].max() + 0.25
		y_min, y_max = X[:,1].min() - 0.25, X[:,1].max() + 0.25
		xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
		Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.35)
		svs = svc.support_vectors_
		svl = plt.plot(svs.T[0], svs.T[1], marker='o', ls='none', ms='15', mfc='none', mec='b', mew=0.5, label="Support vectors")

		dl = plt.scatter(X[:,0], X[:,1], s=30, c=data, cmap="plasma", label="Data")

		x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
		x2 = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
		X1, X2 = np.meshgrid(x1, x2)
		vals = np.zeros(X1.shape)
		for i in range(X1.shape[1]):
			this_x = np.c_[X1[:, i], X2[:, i]]
			vals[:, i] = model.predict(self.gaussianKernelGramMatrix(this_x, X))

		dec = plt.contour(X1, X2, vals, colors="blue", levels=[0,0], label="Decision boundary")
		handlers, labels = ax.get_legend_handles_labels()
		hl = sorted(zip(handlers, labels), key=operator.itemgetter(1))
		h, l = zip(*hl)
		plt.legend(h, l, loc=2, framealpha=0.45)
		# plt.xlim(0, 1)
		# plt.ylim(0.4, 1)
		plt.xlim(-0.6, 0.3)
		plt.ylim(-0.8, 0.6)

		plt.show()

	def dataset3Params(self, X, y, Xval, yval):

		arr_C = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
		arr_sigma = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])

		final_C = None
		final_sigma = None
		b_score = None

		for C in arr_C:
			for Sigma in arr_sigma:

				svc = svm.SVC(C=C, gamma=Sigma)
				svc.fit(X, y)
				score = svc.score(Xval, yval)

			if b_score == None or score > b_score:
				b_score = score
				final_C = C
				final_sigma = Sigma

		return (final_C, final_sigma)





def main():
	
	"""
	Part one- load and plot data
	"""
	curr_path = os.getcwd()
	path_data1 = curr_path + "/data/ex6data1.mat"
	path_data2 = curr_path + "/data/ex6data2.mat"
	path_data3 = curr_path + "/data/ex6data3.mat"

	svm = SVM(path_data1, path_data2, path_data3)
	svm.plotData(svm.X, svm.y)

	"""
	Part two- Training Linear SVM
	"""

	C = 1
	model = svm.svmTrain(svm.X, svm.y, C)
	svm.plotVisualizeBoundary(svm.X, svm.y, model, C)
	C_100 = 100
	model_100 = svm.svmTrain(svm.X, svm.y, C=C_100)
	svm.plotVisualizeBoundary(svm.X, svm.y, model_100, C_100)

	"""
	Part 3: Implement Gaussian Kernel
	"""

	x1 = np.array([1,2,1])
	x2 = np.array([0,4,-1])
	sigma = 2
	sim = svm.gaussianKernel(x1, x2, sigma)
	print(sim)

	"""
	Part 4: load (done above) and visualise data
	"""

	svm.plotData(svm.X1, svm.y1)

	"""
	Part 5: Training SVM with RBF Kernel (Dataset 2)
	"""

	C = 1
	sigma = 0.1
	model = svm.svmTrain(svm.X1, svm.y1, C, tol=0.001, max_passes=-1, type_kernel="precomputed", sigma=sigma)
	svm.plotVisualizeBoundaryMyGaussian(svm.X1, svm.y1, model)

	"""
	Part 6: load and visualise data
	"""
	svm.plotData(svm.X2, svm.y2)

	"""
	Part 7: Training SVM with RBF Kernel
	"""
	C, sigma = svm.dataset3Params(svm.X2, svm.y2, svm.X2val, svm.y2val)
	model = svm.svmTrain(svm.X2, svm.y2, C=C, type_kernel="precomputed", sigma=sigma)
	svm.plotVisualizeBoundaryMyGaussian(svm.X2, svm.y2, model)


if __name__ == "__main__":
	main()