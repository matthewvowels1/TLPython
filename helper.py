
import pandas as pd
import numpy as np
from super_learner import*
from scipy.special import  expit
import random



def generate_data(N, outcome_type, treatment_type):
	''' Generates sample data for evaluating TLP
	::param N: desired sample size (int)
	::param outcome_type: 'cls' or 'reg' for binary or continuous outcome
	::param treatment_type: 'multigroup' or 'binary' for 4-way categorical or binary treatment type'
	::returns data: pd.DataFrame with all data in.'''

	W1 = np.random.binomial(1, 0.5, N)
	W2 = np.random.binomial(1, 0.65, N)
	W3 = np.round(np.random.uniform(0, 4, N), decimals=3)
	W4 = np.round(np.random.uniform(0, 5, N), decimals=3)

	if treatment_type == 'binary':
		Ap = expit(-0.4 + 0.2 * W2 + 0.15 * W3 + 0.2 * W4 + 0.15 * W2 * W4)
		A = np.random.binomial(1, Ap, N)

	elif treatment_type == 'multigroup':
		Ap1 = expit(-0.4 + 0.2 * W2 + 0.3 * W3 + 0.1 * W4 + 0.4 * W2 * W4)
		Ap2 = expit(-0.4 + 0.5 * W2 + 0.1 * W3 + 0.2 * W4 + 0.1 * W2 * W4)
		Ap3 = expit(-0.4 + 0.7 * W2 + 0.5 * W3 + 0.3 * W4 + 0.2 * W2 * W4)
		Ap4 = expit(-0.4 + 0.1 * W2 + 0.2 * W3 + 0.4 * W4 + 0.1 * W2 * W4)
		Ap = np.array([Ap1, Ap2, Ap3, Ap4]).T
		Ap = (Ap / Ap.sum(1).reshape(-1, 1))

		l = [0, 1, 2, 3]
		A = []
		for i in range(len(Ap)):
			ap = Ap[i]
			onehot = np.zeros((4, 1))
			choice = random.choices(l, ap)
			onehot[choice] = 1
			A.append(onehot)
		A = np.concatenate(A, 1).T

	if outcome_type == 'cls':

		if treatment_type == 'binary':
			Y1p = expit(-1 + 1 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4)
			Y0p = expit(-1 + 0 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4)
			Y1 = np.random.binomial(1, Y1p, N)
			Y0 = np.random.binomial(1, Y0p, N)
			Y = Y1 * A + Y0 * (1 - A)

		if treatment_type == 'multigroup':
			Y3p = expit(-1 + 1 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4)
			Y2p = expit(-1 + 2 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4)
			Y1p = expit(-1 + 0.5 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4)
			Y0p = expit(-1 + 8 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4)

			Y3 = np.random.binomial(1, Y3p, N)
			Y2 = np.random.binomial(1, Y2p, N)
			Y1 = np.random.binomial(1, Y1p, N)
			Y0 = np.random.binomial(1, Y0p, N)

			Y = A[:, 3] * Y3 + A[:, 2] * Y2 + A[:, 1] * Y1 + A[:, 0] * Y0

	elif outcome_type == 'reg':

		if treatment_type == 'binary':
			Y1 = -1 + 2 * A - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4
			Y0 = -1 + 0 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4

			Y = Y1 * A + Y0 * (1 - A)

		if treatment_type == 'multigroup':
			Y3 = -1 + 1 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4
			Y2 = -1 + 2 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4
			Y1 = -1 + 0.5 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4
			Y0 = -1 + 3 - 0.1 * W1 + 0.3 * W2 + 0.25 * W3 + 0.2 * W4 + 0.15 * W2 * W4

			Y = A[:, 3] * Y3 + A[:, 2] * Y2 + A[:, 1] * Y1 + A[:, 0] * Y0
		Y = np.clip(Y, -1, 6)

	if treatment_type == 'multigroup':
		data = pd.DataFrame([W1, W2, W3, W4, A, Y, Y0, Y1, Y2, Y3]).T
		data.columns = ['W1', 'W2', 'W3', 'W4', 'Adummy', 'Y', 'Y0', 'Y1', 'Y2', 'Y3']

		A_group = []
		for a in data.Adummy.values:
			val = np.where(a == 1)
			A_group.append(val[0])
		A_group = np.concatenate(A_group)
		data['A'] = A_group
	elif treatment_type == 'binary':
		data = pd.DataFrame([W1, W2, W3, W4, A, Y, Y1, Y0]).T
		data.columns = ['W1', 'W2', 'W3', 'W4', 'A', 'Y', 'Y1', 'Y0']
	return data
