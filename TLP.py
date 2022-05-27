import numpy as np
import os
import pandas as pd
from super_learner import*
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import logit, expit
import random

class TLP(object):
	''' Targeted Learning class.
	::param data: a pandas dataframe with outcome, treatment, and covariates
	::param cause: a string with the dataframe column name for the treatment/cause of interest
	::param outcome: a string with the dataframe column name for the outcome/effect of interest
	::param confs: a list of strings of the dataframe column names for the confounders
	::param precs: a list of strings of the dataframe columns names for precision/risk variables
	::param Q_learners: a list of strings for the abbreviations of the learners to be included in the outcome Q SL
	::param G_learners: a list of strings for the abbreviations of the learners to be included in the treatment G SL
	::param outcome_type: a string 'reg' or 'cls' incicating whether the outcome is binary or continuous
	::param: outcome_upper_bound, outcome_lower_bound: floats for the upper and lower bound of the outcomes for rescaling to [0,1]
	'''

	def __init__(self, data, cause, outcome, confs, precs, Q_learners, G_learners, outcome_type='reg',
	             outcome_upper_bound=None, outcome_lower_bound=None):

		# general settings
		self.outcome_type = outcome_type  # reg or cls
		self.outcome_upper_bound = outcome_upper_bound  # for bounded outcomes
		self.outcome_lower_bound = outcome_lower_bound  # for bounded outcomes

		# data and variable names
		self.data = data  # pd.DataFrame()
		self.n = len(self.data)
		self.cause = cause
		self.outcome = outcome
		self.confs = confs
		self.precs = precs

		self.Q_X = self.data[list(set(confs)) + list(set(precs)) + [cause]]
		self.G_X = self.data[list(set(confs))]
		self.num_confs = len(list(set(confs)))
		self.Q_Y = self.data[outcome].astype('int') if outcome_type == 'cls' else self.data[outcome]
		self.G_Y = self.data[cause]

		self.A = self.data[cause]
		self.A_dummys = self.A.copy()
		self.A_dummys = self.A_dummys.astype('category')
		self.A_dummys = pd.get_dummies(self.A_dummys, drop_first=False)

		if (self.outcome_upper_bound is not None) and (self.outcome_type == 'reg'):
			self.Q_Y = (self.Q_Y - self.outcome_lower_bound) / (self.outcome_upper_bound - self.outcome_lower_bound)

		self.groups = np.unique(self.A)

		# Super Learners:
		self.Q_learners = Q_learners  # list of learners
		self.G_learners = G_learners  # list of learners
		self.gslr = None
		self.qslr = None

		self.Qbeta = None  # beta learner weights for Q model
		self.Gbeta = None  # beta learner weights for G model

		# targeting and prediction storage
		self.Gpreds = None
		self.QAW = None
		self.Qpred_groups = {}
		self.Gpreds = None
		self.first_estimates = {}
		self.first_effects = {}
		self.updated_estimates = {}
		self.updated_effects = {}
		self.updated_estimates_dr = {}
		self.updated_effects_dr = {}
		self.clev_covs = {}
		self.clev_covs_dr = {}
		self.epsilons = {}
		self.ses = {}
		self.ses_dr = {}
		self.ps = {}
		self.ps_dr = {}

		self.condition1s = []
		self.condition2s = []
		self.condition3s = []

		self.r_or = {}
		self.r_ps1 = {}
		self.r_ps2 = {}


	def fit(self, k, standardized_outcome=False, calibrationQ=False, calibrationG=False):
		'''Fits the superlearners
		::param k: the number of folds to use in k-fold cross-validation
		::param standardized_outcome: whether to standardize the outcome (only applicable to continuous outcomes)
		::param calibration: whether to calibrate the classifiers (not applicable to regressors)
		::returns all_preds_Q, gts_Q, all_preds_G, gts_G: arrays of test-fold predictions and GT for Q and G models
		'''


		print('Training Q Learners...')
		self.qslr = SuperLearner(output=self.outcome_type, calibration=calibrationQ, learner_list=self.Q_learners, k=k,
		                         standardized_outcome=standardized_outcome)
		all_preds_Q, gts_Q = self.qslr.fit(x=self.Q_X, y=self.Q_Y)

		# QAW PREDS
		print('Generating QAW Predictions ')
		QAW = self.qslr.predict(self.Q_X)[:, 0] if self.outcome_type == 'reg' else self.qslr.predict_proba(
			self.Q_X)[:, 1]
		self.QAW = np.clip(QAW, 0.005, 0.995) if (self.outcome_type == 'cls') or (self.outcome_upper_bound is not None) else QAW

		if self.outcome_upper_bound is not None and self.outcome_type == 'reg':
			print('Bounding outcome predictions.')
			self.QAW = np.clip(self.QAW, 0.005, 0.995)

		all_preds_G, gts_G = None, None
		# PROPENSITY SCORES
		if self.num_confs != 0:
			print('Training G Learners...')
			self.gslr = SuperLearner(output='proba', calibration=calibrationG, learner_list=self.G_learners, k=k,
			                         standardized_outcome=False)
			all_preds_G, gts_G = self.gslr.fit(x=self.G_X, y=self.G_Y)
			print('Generating G Predictions ')
			self.Gpreds = np.clip(self.gslr.predict_proba(self.G_X), 0.005, 0.995)

			print('SuperLearner Training Completed.')
			self.Qbeta = self.qslr.beta
			self.Gbeta = self.gslr.beta

		else:
			print('No confounders, computing marginal probabilities of treatment.')
			if (self.G_Y, pd.DataFrame) or isinstance(self.G_Y, pd.DataFrame):
				unique, counts = np.unique(self.G_Y.values, return_counts=True)

			else:
				unique, counts = np.unique(self.G_Y, return_counts=True)

			self.Gpreds = np.repeat(np.clip(counts/len(self.G_Y), 0.005, 0.995).reshape(1, -1), len(self.G_Y), axis=0)

		return all_preds_Q, gts_Q, all_preds_G, gts_G

	def _fit_or_ps(self, k, preds, outcome_type, targets):
		'''Fits the superlearners for R-PS1, and R-PS2'''

		r_slr = SuperLearner(output=outcome_type, learner_list=['LR'], k=k, standardized_outcome=False)
		_, _ = r_slr.fit(x=preds, y=targets)
		r_slr_preds = r_slr.predict(preds)[:, 0] if outcome_type == 'reg' else r_slr.predict_proba(preds)[:, 1]
		r_slr_preds = r_slr_preds if (outcome_type != 'proba') else np.clip(r_slr_preds, 0.005, 0.995)
		return r_slr_preds


	def dr_target_multigroup(self, group_comparisons=None, iterations=10, k=5):
		assert self.num_confs != 0, 'No confounders, use regular target_multigroup() function instead.'
		# GO THROUGH REGULAR UPDATE PROCESS BEFORE COMPUTING ADDITIONAL ELEMENTS FOR THE DOUBLY ROBUST INFERENCE
		# STEP 1 in Benkeser et al. 2017
		print('Generating predictions for counterfactual outcomes...')
		self._q_pred_groups()

		for group_comparison in group_comparisons:
			group_a = group_comparison[0]
			group_ref = group_comparison[1]
			dummys = self.A_dummys.iloc[:, group_a].values
			G_a = self.Gpreds[:, group_a]
			Q_group_a = self.Qpred_groups[group_a]
			Q_group_ref = self.Qpred_groups[group_ref]
			QAW_st = self.QAW
			Y = self.Q_Y

			for j in range(iterations):  # in Benkeser et al. (2017), this is k, not j
				print('Doubly-robust inference iteration:', j + 1)
				# STEP 2 in Benkeser et al. 2017
				H1nk = dummys / G_a

				if self.outcome_type == 'cls' or self.outcome_upper_bound is not None:
					eps = sm.GLM(np.asarray(Y[dummys == 1]).astype('float'), H1nk[dummys == 1],
					             offset=logit(Q_group_a[dummys == 1]),
					             family=sm.families.Binomial()).fit().params[0]
				else:
					eps = (sm.GLM(np.asarray(Y[dummys == 1]).astype('float'), H1nk[dummys == 1],
					              offset=Q_group_a[dummys == 1]).fit()).params[
						0]

				if self.outcome_type == 'cls' or self.outcome_upper_bound is not None:
					Q_group_a = (expit(logit(Q_group_a) + eps * (1 / G_a)))

				else:
					Q_group_a = Q_group_a + eps * (1 / G_a)

				# STEP 3 in Benkeser et al. 2017
				targets_rps2_1 = (dummys - G_a) / G_a
				targets_rps1_1 = dummys

				g10r_1 = self._fit_or_ps(k=k, preds=Q_group_a.reshape(-1, 1), outcome_type='proba',
				                            targets=targets_rps1_1)
				g20r_1 = self._fit_or_ps(k=k, preds=Q_group_a.reshape(-1, 1), outcome_type='reg',
				                            targets=targets_rps2_1)

				# STEP 4 in Benkeser et al. 2017
				H2nk = dummys * g20r_1 / g10r_1

				if self.outcome_type == 'cls' or self.outcome_upper_bound is not None:
					eps_ps = sm.GLM(np.asarray(Y[dummys == 1]).astype('float'), H2nk[dummys == 1],
					                offset=logit(Q_group_a[dummys == 1]),
					                family=sm.families.Binomial()).fit().params[0]
				else:
					eps_ps = (sm.GLM(np.asarray(Y[dummys == 1]).astype('float'), H2nk[dummys == 1],
					                 offset=Q_group_a[dummys == 1]).fit()).params[0]

				if self.outcome_type == 'cls' or self.outcome_upper_bound is not None:
					Q_group_a = (expit(logit(Q_group_a) + eps_ps * H2nk))

				else:
					Q_group_a = Q_group_a + eps_ps * H2nk

				# STEP 5 in Benkeser et al. 2017
				residual = Y - Q_group_a

				Q0r_1 = self._fit_or_ps(k=5, preds=G_a.reshape(-1, 1), outcome_type='reg',
				                           targets=residual)

				# STEP 6 in Benkeser et al. 2017
				H3nk = Q0r_1 / G_a

				eps_or = sm.GLM(dummys.astype('float'), H3nk, offset=logit(G_a),
				                family=sm.families.Binomial()).fit().params[0]

				G_a = np.clip((expit(logit(G_a) + eps_or * H3nk)), 0.005, 0.995)

				# finally reupdate targeted estimate
				H1nk1 = dummys / G_a
				H1nk0 = - dummys / (1 - G_a)
				H1nk = H1nk1 + H1nk0

				if self.outcome_type == 'cls' or self.outcome_upper_bound is not None:
					eps = sm.GLM(np.asarray(Y).astype('float'), H1nk, offset=logit(QAW_st),
					             family=sm.families.Binomial()).fit().params[0]
				else:
					eps = (sm.GLM(np.asarray(Y).astype('float'), H1nk[dummys == 1], offset=QAW_st).fit()).params[
						0]

				if self.outcome_type == 'cls' or self.outcome_upper_bound is not None:
					Q_group_a = (expit(logit(Q_group_a) + eps * H1nk1))
					Q_group_ref = (expit(logit(Q_group_ref) + eps * H1nk0))
					QAW_st = (expit(logit(QAW_st) + eps * H1nk))
				else:
					Q_group_a = Q_group_a + eps * H1nk1
					Q_group_ref = Q_group_ref + eps * H1nk0
					QAW_st = QAW_st + eps * H1nk

				self.condition1s.append((((dummys * g20r_1) / g10r_1) * (
							Y - QAW_st)).mean())  # ('8') in R tutorial, p 870 in Benkeser et al. 2017
				self.condition2s.append(
					((Q0r_1 / G_a) * (dummys - G_a)).mean())  # (6) in R tutorial, p 869 in Benkeser et al. 2017
				self.condition3s.append(((dummys / G_a) * (Y - QAW_st)).mean())  # (5) in R tutorial

			# update self.(all_variables)
			self.updated_estimates_dr[str(group_comparison)] = (Q_group_a, Q_group_ref, QAW_st)
			self.updated_effects_dr[str(group_comparison)] = (Q_group_a - Q_group_ref).mean()
			self.clev_covs_dr[str(group_comparison)] = (H1nk1, H1nk0, H1nk)

		self._computing_IF(group_comparisons=group_comparisons, dr_flag=True)

		return self.first_effects, self.updated_effects_dr, self.ses_dr, self.ps_dr

	def target_multigroup(self, group_comparisons=None):
		'''Runs multigroup targeted learning to estimate causal effects, standard errors, and p-values.
		::param group_comparisons: a list of lists e.g. [[2, 1], [3, 1]] for comparing groups 2 vs 1, and 3 vs 1. These
		groups MUST correspond with the treatment groups in the dataframe!
		::returns first_effects, updated_effects, ses, ps: arrays for pre-update & post-update effects, standard errors
		and p values.
		'''

		print('Generating predictions for counterfactual outcomes...')
		self._q_pred_groups()

		print('Computing group differences between counterfactual outcomes...')
		self._group_diffs(group_comparisons=group_comparisons)

		print('Computing clever covariates...')
		self._q_clever_covs(group_comparisons=group_comparisons)

		print('Estimating fluctuation parameters...')
		self._fluctation_params(group_comparisons=group_comparisons)

		print('Updating initial counterfactual predictions...')
		self._updating_q(group_comparisons=group_comparisons)

		print('Deriving the Influence Function, standard error, CI bounds and p-values')
		self._computing_IF(group_comparisons=group_comparisons, dr_flag=False)

		return self.first_effects, self.updated_effects, self.ses, self.ps


	def _inference(self, ic, effect):
		IC_var = np.var(ic, ddof=1)
		se = (IC_var / self.n) ** 0.5
		p = 2 * (1 - norm.cdf(np.abs(effect) / se))
		upper_bound, lower_bound = (effect + 1.96 * se), (effect - 1.96 * se)
		return se, p, upper_bound, lower_bound

	def _q_pred_groups(self):
		# INTERVENTIONAL Y PREDS
		for group in self.groups:
			int_data = self.Q_X.copy()
			int_data[self.cause] = group
			qps = self.qslr.predict(int_data)[:,
			                           0] if self.outcome_type == 'reg' else self.qslr.predict_proba(int_data)[:, 1]

			self.Qpred_groups[group] = np.clip(qps, 0.005, 0.995) if (self.outcome_type == 'cls') or (self.outcome_upper_bound is not None) else qps

	def _group_diffs(self, group_comparisons):
		# GROUP DIFFERENCES
		for group_comparison in group_comparisons:
			group_a = group_comparison[0]
			group_ref = group_comparison[1]

			group_a_preds = self.Qpred_groups[group_a]
			group_ref_preds = self.Qpred_groups[group_ref]

			difference = (group_a_preds - group_ref_preds)
			self.first_estimates[str(group_comparison)] = difference


	def _q_clever_covs(self, group_comparisons):
		# CLEVER COVARIATES
		for group_comparison in group_comparisons:
			group_a = group_comparison[0]
			group_a_inv_prop = 1 / self.Gpreds[:, group_a]
			group_not_a_inv_prop = - 1 / (1 - self.Gpreds[:, group_a])
			group_clev_cov = ((self.A_dummys.iloc[:, group_a] / self.Gpreds[:, group_a]) - (
						1 - self.A_dummys.iloc[:, group_a]) / (1 - self.Gpreds[:, group_a])).values
			self.clev_covs[str(group_comparison)] = (group_clev_cov, group_a_inv_prop, group_not_a_inv_prop)


	def _fluctation_params(self, group_comparisons):
		# ESTIMATE FLUCTUATION PARAMETERS
		for group_comparison in group_comparisons:
			group_clev_cov = self.clev_covs[str(group_comparison)][0]
			if self.outcome_type == 'cls' or self.outcome_upper_bound is not None:
				eps = sm.GLM(np.asarray(self.Q_Y).astype('float'), group_clev_cov, offset=logit(self.QAW),
				             family=sm.families.Binomial()).fit().params[0]
			else:
				eps = (sm.GLM(np.asarray(self.Q_Y).astype('float'), group_clev_cov, offset=self.QAW).fit()).params[0]

			self.epsilons[str(group_comparison)] = eps


	def _updating_q(self, group_comparisons):
		# UPDATING PREDICTIONS
		for group_comparison in group_comparisons:
			group_a = group_comparison[0]
			group_ref = group_comparison[1]
			group_a_orig = self.Qpred_groups[group_a]
			group_ref_orig = self.Qpred_groups[group_ref]
			group_aw_orig = self.QAW

			self.first_effects[str(group_comparison)] = group_a_orig.mean() - group_ref_orig.mean()

			eps = self.epsilons[str(group_comparison)]
			clev_cov_AW = self.clev_covs[str(group_comparison)][0]
			clev_cov_a = self.clev_covs[str(group_comparison)][1]
			clev_cov_ref = self.clev_covs[str(group_comparison)][2]

			if self.outcome_type == 'cls' or self.outcome_upper_bound is not None:
				group_a_update = (expit(logit(group_a_orig) + eps * clev_cov_a))
				group_ref_update = (expit(logit(group_ref_orig) + eps * clev_cov_ref))
				group_aw_update = (expit(logit(group_aw_orig) + eps * clev_cov_AW))
			else:
				group_a_update = group_a_orig + eps * clev_cov_a
				group_ref_update = group_ref_orig + eps * clev_cov_ref
				group_aw_update = group_aw_orig + eps * clev_cov_AW

			self.updated_estimates[str(group_comparison)] = (group_a_update, group_ref_update, group_aw_update)
			self.updated_effects[str(group_comparison)] = (group_a_update - group_ref_update).mean()

	def _computing_IF(self, group_comparisons, dr_flag=False):
		# COMPUTING THE IF
		for group_comparison in group_comparisons:
			clev_cov_group = self.clev_covs[str(group_comparison)][0] if not dr_flag else self.clev_covs_dr[str(group_comparison)][0]
			ystar_a = self.updated_estimates[str(group_comparison)][0] if not dr_flag else self.updated_estimates_dr[str(group_comparison)][0]
			ystar_ref = self.updated_estimates[str(group_comparison)][1] if not dr_flag else self.updated_estimates_dr[str(group_comparison)][1]
			QAW_st = self.updated_estimates[str(group_comparison)][2]  if not dr_flag else self.updated_estimates_dr[str(group_comparison)][2]
			effect_star = self.updated_effects[str(group_comparison)] if not dr_flag else self.updated_effects_dr[str(group_comparison)]
			IC = clev_cov_group * (self.Q_Y.values - QAW_st) + (ystar_a - ystar_ref) - effect_star
			se, p, upper_bound, lower_bound = self._inference(ic=IC, effect=effect_star)
			if dr_flag:
				self.ses_dr[str(group_comparison)] = (se, upper_bound, lower_bound)
				self.ps_dr[str(group_comparison)] = p
			else:
				self.ses[str(group_comparison)] = (se, upper_bound, lower_bound)
				self.ps[str(group_comparison)] = p