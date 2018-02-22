import os
import os.path
import operator

import re

import nltk
from nltk import PorterStemmer
from nltk.stem.porter import PorterStemmer

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

class SVMSpam():

	def __init__(self, path_vocablist, pathSpamTrain):
		
		try:

			with open(path_vocablist,'r') as vocabFile:

				# Store all dictionary words in dictionary vocabList
				self.vocabList = {}
				for line in vocabFile.readlines():
					i, word = line.split()
					self.vocabList[word] = int(i)

		except IOError:
			print("No such file")
			raise

		try:

			self._spamTrain = loadmat(pathSpamTrain)
			self.X = self._spamTrain['X']
			self.y = self._spamTrain['y']

		except IOError:
			print("No pathSpamTrain.mat file")
			raise


	def getVocabList(self):

		return (self.vocabList)

	def readFile(self, filePath):

		try:
			with open(filePath, 'r') as openFile:
				file_content = openFile.read()
		except:
			file_content = ''
			print("Unable tpo open {:s}".format(filePath))

		return (file_content)

	def processEmail(self, email_contents):

		"""
		PROCESSEMAIL- preprocesses the body of an email and returns a list of word_indices
		word_indices = PROCESSEMAIL(email_contents) preprocesses the body of an email and returns 
		a list of indicies of the words contained in the email.
		"""

		# get Vocabulary
		vocabList = self.getVocabList()

		word_indices = []

		# ========================= Preprocess Email ==========================================

		"""
		Find the Headers (\n\n and remove)
		Uncomment the following lines if you are working with raw emails with the full headers
		"""
		# hdstart = email_contents
		# if hdstart:
		#	email_contents = email_contents[hdstart:]

		# All to lower case
		email_contents = email_contents.lower()

		"""
		Strip all HTML
		Looks for any expression that starts with < and end with 
		(it doesn't have any < or > in the tag)> and replace it on whitespace
		"""

		email_contents = re.sub('<[^<>]+>', ' ', email_contents)

		"""
		Handle numbers (Normilizing numbers)
		Look for one or more charecters between 0-9
		"""

		email_contents = re.sub('[0-9]+', "number", email_contents)

		"""
		Handle URLs (Normilizing URLs)
		Look for strings starting with http:// or https:// and repalce on httpaddr
		"""

		email_contents = re.sub('(http|https)://[^\s]*', "httpaddr", email_contents)

		"""
		Handle Email Address (Normilizing Email Addresses)
		Look for strings with @ in the middle 
		"""

		email_contents = re.sub('[^s\s]+@[^\s]+', "emailaddr", email_contents)

		"""
		Handle $ sign (Normilizing Dollars)
		"""

		email_contents = re.sub('[$]+', "dollar", email_contents)

		# ============================ Tokenize Email =====================================

		print("===== Email Processed =====")

		l = 0

		# Slightly different order from matlab version

		# Split and also get rid of any punctuation
		# regex may need further debugging...

		#.∧＿∧ 
		#( ･ω･｡)つ━☆・*。 
		#⊂  ノ    ・゜+. 
		#しーＪ   °。+ *´¨) 
		# .· ´¸.·*´¨) 
		#  (¸.·´ (¸.·'* Wow such magic (no)
		email_contents = re.split(r'[@$/#.-:&\*\+=\[\]?!(){},\'\'\">_<;%\s\n\r\t]+', email_contents)
		
		for token in email_contents:

			# 	# Remove any non alphanumeric characters
			token = re.sub("[^a-zA-Z0-9]", '', token)

			# Stem the word

			token = PorterStemmer().stem(token.strip())
			# token = PorterStemmer().stem_word(token.strip())

			if len(token) < 1:
				continue

			indx = self.vocabList[token] if token in self.vocabList else 0

			if indx > 0:
				word_indices.append(indx)

			# print(token)

		print("\n\n================================\n")

		return (word_indices)

	def emailFeatures(self, word_indices):

		"""
		Takes in a word_indices vector and produce a feature vector from the word indices
		"""
		n = len(self.getVocabList())

		x = np.zeros((n, 1))

		for i in word_indices:
			x[i] = 1

		return (x)

	def svmTrain(self, X, y, C=0.1, tol=1e-3, max_passes=-1, sigma=0.1):

		y = y.flatten()

		model = svm.SVC(C=C, kernel="linear", tol=tol, max_iter=max_passes, verbose=2)
		print(X)
		model.fit(X, y)

		return (model)


def main():
	
	"""
	# =================== Part 1 (Eamil preprocessing) and Part 2 (Feature Extraction) =================
	"""
	path_vocablist = os.getcwd() + "/data/vocab.txt"
	path_email1 = os.getcwd() + "/data/emailSample1.txt"
	path_email2 = os.getcwd() + "/data/emailSample2.txt"
	path_email3 = os.getcwd() + "/data/emailSample3.txt"
	pathSpamTrain = os.getcwd() + "/data/spamTrain.mat"

	svmSpam = SVMSpam(path_vocablist, pathSpamTrain)
	# print(svmSpam.vocabList)
	
	# Extract features

	file_content = svmSpam.readFile(path_email1)
	word_indices = svmSpam.processEmail(file_content)
	features = svmSpam.emailFeatures(word_indices)

	# Print stats
	print("Length of feature vector: {:d}".format( len(features) ))
	print("Number of non-zero entries: {:d}".format( np.sum(features > 0) ))

	# input("Program Paused. Press Enter to continue.")

	"""
	# ========================= Part 3: Train Linear SVM for Spam Classification ==================
	"""

	C = 0.1
	model = svmSpam.svmTrain(svmSpam.X, svmSpam.y, C)

	"""
	# ======================= Part 4: Test Spam Classificator =====================================
	"""
	predict = model.predict(svmSpam.X)

	print("Training accuranct: {:f}".format(np.mean((predict == svmSpam.y).astype(int)) * 100))

	"""
	# ====================== Part 5: Top Prediciton Spam ==========================================
	"""

	w = model.coef_[0]
	print(w)
	print(w.argsort())
	indices = w.argsort()[::-1][:15]

	vocablist = sorted(svmSpam.getVocabList().keys())

	print("\nTop predictors of spam: \n")
	for i in indices:
		print(" {:s} ({:f})".format(vocablist[i], float(w[i])))

	"""
	# Part 6: Try Your Own Emails
	"""

	file_content = svmSpam.readFile(path_email1)
	word_indices = svmSpam.processEmail(file_content)
	features = svmSpam.emailFeatures(word_indices)

	predict = model.predict(features.reshape(1, -1))

	print("(1 indicates as spam, 0 indicates as not spam)")
	print("\nProcessed {:s}\n\nSpam Classification: {:s}\n".format(path_email1, str(predict)))

	file_content = svmSpam.readFile(path_email2)
	word_indices = svmSpam.processEmail(file_content)
	features = svmSpam.emailFeatures(word_indices)

	predict = model.predict(features.reshape(1, -1))

	print("(1 indicates as spam, 0 indicates as not spam)")
	print("\nProcessed {:s}\n\nSpam Classification: {:s}\n".format(path_email2, str(predict)))

	file_content = svmSpam.readFile(path_email3)
	word_indices = svmSpam.processEmail(file_content)
	features = svmSpam.emailFeatures(word_indices)

	predict = model.predict(features.reshape(1, -1))

	print("(1 indicates as spam, 0 indicates as not spam)")
	print("\nProcessed {:s}\n\nSpam Classification: {:s}\n".format(path_email3, str(predict)))



if __name__ == "__main__":
	main()