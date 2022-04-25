import numpy as np
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures#
import pandas as pd


def fn(x, A, b):
	return np.linalg.norm(A.dot(x) - b)


''' An example dictionary of estimators can be specified as follows:
Gest_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True),
                                 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                                 'AB': AdaBoostClassifier(), 'poly': 'poly'}

Note that 'poly' is specified as a string because there are no default polynomial feature regressors in sklearn.
This one defaults to 2nd order features (e.g. x1*x2, x1*x3 etc...)'''


def combiner_solve(x, y):
	# adapted from https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1/33388181
	beta_0, rnorm = nnls(x, y)
	cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
	bounds = [[0.0, None]] * x.shape[1]
	minout = minimize(fn, beta_0, args=(x, y), method='SLSQP', bounds=bounds, constraints=cons)
	beta = minout.x
	return beta


class SuperLearner(object):
	def __init__(self, output, est_dict, k, standardized_outcome=False):

		self.k = k  # number of cross validation folds
		self.beta = None
		self.output = output  # 'reg' for regression, 'proba' or 'cls' classification
		self.trained_superlearner = None
		self.est_dict = est_dict  # dictionary of learners/algos
		self.standardized_outcome = standardized_outcome

		self.x_std = None
		self.x_mean = None
		self.y_std = None
		self.y_mean = None
		self.num_classes = None

	def fit(self, x, y):
		x = x.values if isinstance(x, pd.DataFrame) else x
		y = y.values[:, 0] if isinstance(y, pd.DataFrame) else y

		# mean and std for full dataset (can be reused wth new data at prediction time)
		self.x_std = x.std(0)
		self.x_mean = x.mean(0)

		if self.standardized_outcome:
			self.y_std = y.std(0)
			self.y_mean = y.mean(0)

		if (self.output == 'cls') or (self.output == 'proba'):
			self.num_classes = np.unique(y)

			if len(self.num_classes) == 2:
				self.num_classes = 1
			elif len(self.num_classes) > 2:
				self.num_classes = len(self.num_classes)
				self.output = 'cat'
		else:
			self.num_classes = 1

		kf = KFold(n_splits=self.k, shuffle=True, random_state=0)

		all_preds = np.zeros((len(y), len(self.est_dict)))  # for test preds

		i = 0
		for key in self.est_dict.keys():
			print('Training estimator:', key)

			est = self.est_dict[key]

			preds = []
			gts = []

			for train_index, test_index in kf.split(x):
				x_train = x[train_index]
				x_test = x[test_index]
				y_train = y[train_index]
				y_test = y[test_index]

				# per train/test fold means and standard deviations
				x_std = x_train.std(0)
				x_mean = x_train.mean(0)
				x_train = (x_train - x_mean) / x_std
				x_test = (x_test - x_mean) / x_std

				if self.standardized_outcome:
					y_std = y_train.std(0)
					y_mean = y_train.mean(0)
					y_train = (y_train - y_mean) / y_std
					y_test = (y_test - y_mean) / y_std

				if key == 'poly':
					est = LogisticRegression(C=1e2, max_iter=350) if ((self.output == 'cls') or (
							self.output == 'proba') or (self.output == 'cat')) else LinearRegression()
					poly = PolynomialFeatures(2)
					x_train_poly = poly.fit_transform(x_train)
					x_test_poly = poly.fit_transform(x_test)

					est.fit(x_train_poly, y_train)

				else:
					est.fit(x_train, y_train)

				p = est.predict(x_test_poly) if key == 'poly' else est.predict(x_test)
				preds.append(p)
				gts.append(y_test)

			preds = np.concatenate(preds)
			gts = np.concatenate(gts)

			all_preds[:, i] = preds

			i += 1

		# estimate betas on test predictions
		self.beta = combiner_solve(all_preds, gts)  # all_preds is of shape [batch, categories, predictors]

		# now train each estimator on full dataset

		x = (x - self.x_mean) / self.x_std
		if self.standardized_outcome:
			y = (y - self.y_mean) / self.y_std

		for key in self.est_dict.keys():
			print('Training estimator on full data:', key)

			est = self.est_dict[key]

			if key == 'poly':
				est = LogisticRegression(C=1e2, max_iter=350) if ((self.output == 'cls') or (
						self.output == 'proba') or (self.output == 'cat')) else LinearRegression()
				poly = PolynomialFeatures(2)
				x_poly = poly.fit_transform(x)

				est.fit(x_poly, y)

			else:
				est.fit(x, y)

			self.est_dict[key] = est

	def predict(self, x):
		x = x.values if isinstance(x, pd.DataFrame) else x
		x_ = (x - self.x_mean) / self.x_std
		all_preds = np.zeros((len(x_), len(self.est_dict)))
		i = 0

		for key in self.est_dict.keys():
			est = self.est_dict[key]
			if key == 'poly':
				poly = PolynomialFeatures(2)
				x_scaled = poly.fit_transform(x_)

			preds = est.predict(x_) if key != 'poly' else est.predict(x_scaled)

			all_preds[:, i] = preds

			i += 1

		weighted_preds = np.dot(all_preds, self.beta)
		weighted_preds = weighted_preds.reshape(-1, 1)
		if self.standardized_outcome:
			weighted_preds = (weighted_preds * self.y_std) + self.y_mean
		return weighted_preds

	def predict_proba(self, x):
		x = x.values if isinstance(x, pd.DataFrame) else x

		x_ = (x - self.x_mean) / self.x_std

		all_preds = np.zeros((len(x), self.num_classes, len(self.est_dict)))
		i = 0

		for key in self.est_dict.keys():
			est = self.est_dict[key]
			if key == 'poly':
				poly = PolynomialFeatures(2)
				x_poly = poly.fit_transform(x_)

			preds = est.predict_proba(x_) if key != 'poly' else est.predict_proba(x_poly)
			all_preds[:, :, i] = preds

			i += 1

		weighted_preds = []
		for cl in range(self.num_classes):
			preds = np.dot(all_preds[:, cl, :], self.beta)

			if self.standardized_outcome:
				preds = (preds * self.y_std) + self.y_mean

			weighted_preds.append(preds)

		weighted_preds = np.asarray(weighted_preds).T

		return weighted_preds
