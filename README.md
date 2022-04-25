# TLPython
Targeted Learning for Binary and Categorical Treatment


## Example Usage:


```python
from TLP import TLP
from helper import generate_data

# NOTE: THE GROUP NAMES FOR THE TREATMENT MUST BE INDEXED WITH INTEGERS STARTING FROM 0

est_dict_Q = ['Elastic', 'BR', 'SV', 'LR', 'RF', 'MLP', 'AB', 'poly']
est_dict_G = ['LR', 'NB', 'MLP','SV', 'poly', 'RF','AB']

'''
BR = bayesian ridge (reg)
Elastic = elastic net (reg)
SV = support vector (reg/cls)
LR = linear/logistic regression (reg/cls)
RF = random forest (reg/cls)
MLP = multilayer perceptron (reg/cls)
AB = adaboost (reg/cls)
poly = polynomial linear/logistic regression (reg/cls)
NB = Gaussian Naive Bayes (cls)'''

outcome_type = 'reg'   # cls or reg
treatment_type = 'multigroup' # binary or multigroup
N = 600
group_comparisons =[[1,0],[2,0],[3,0]]  # comparison in list format with 'group A [vs] reference_group'
k = 8  # number of folds for SL training


data = generate_data(N=N, outcome_type=outcome_type, treatment_type=treatment_type)  # example function in ipynb

true_psi_1_0 = data.Y1.mean() - data.Y0.mean()
true_psi_2_0 = data.Y2.mean() - data.Y0.mean()
true_psi_3_0 = data.Y3.mean() - data.Y0.mean()

# initialise TLP object
tlp = TLP(data, cause='A', outcome='Y', confs=['W1', 'W2', 'W3', 'W4'],
          precs=[], outcome_type=outcome_type, Q_learners=est_dict_Q, G_learners=est_dict_G)


# fit SuperLearners
all_preds_Q, gts_Q, all_preds_G, gts_G = tlp.fit(k=k, standardized_outcome=False, calibration=True)

# 'do' targeted learning
pre_update_effects, post_update_effects, ses, ps = tlp.target_multigroup(group_comparisons=group_comparisons)

# compare results
print(post_update_effects)
print(true_psi_1_0, true_psi_2_0, true_psi_3_0)
```