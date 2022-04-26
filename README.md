# TLPython
Targeted Learning [1,2,3,4] for estimation of the Average Causal/Treatment Effect (ACE/ATE) for binary or categorical (facilitating multigroup comparison estimates) treatment, for continuous or binary outcomes.

Uses Super Learners [5] for outcome and treatment/nuisance parameter models.


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


### Outstanding Work

- Longitudinal (followin G-computation/estimation) [7,8]
- Highly Adaptive Lasso [6]


### References
[1.] Van der Laan, M. J., & Rose, S. (2011). Targeted learning: causal inference for observational and experimental data (Vol. 4). New York: Springer
[2.] Luque‐Fernandez, M. A., Schomaker, M., Rachet, B., & Schnitzer, M. E. (2018). Targeted maximum likelihood estimation for a binary treatment: A tutorial. Statistics in medicine, 37(16), 2530-2546.
[3.] Coyle, J. R., Hejazi, N. S., Malenica, I., Phillips, R. V., Arnold, B. F., Mertens, A., ... & van der Laan, M. J. (2020). Targeting learning: robust statistics for reproducible research. arXiv preprint arXiv:2006.07333.
[4.] Van Der Laan, M. J., & Rubin, D. (2006). Targeted maximum likelihood learning. The international journal of biostatistics, 2(1).
[5.] Van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. Statistical applications in genetics and molecular biology, 6(1).
[6.] Benkeser, D., & Van Der Laan, M. (2016, October). The highly adaptive lasso estimator. In 2016 IEEE international conference on data science and advanced analytics (DSAA) (pp. 689-696). IEEE.
[7.] Petersen, M., Schwab, J., Gruber, S., Blaser, N., Schomaker, M., & van der Laan, M. (2014). Targeted maximum likelihood estimation for dynamic and static longitudinal marginal structural working models. Journal of causal inference, 2(2), 147-185.
[8.] van der Laan, M. J., & Gruber, S. (2012). Targeted minimum loss based estimation of causal effects of multiple time point interventions. The international journal of biostatistics, 8(1).


