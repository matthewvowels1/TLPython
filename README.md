# TLPython
Targeted Learning [1,2,3,4] for estimation of the Average Causal/Treatment Effect (ACE/ATE) for binary or categorical (facilitating multigroup comparison estimates) treatment, for continuous or binary outcomes.

Uses Super Learners [5] for outcome and treatment/nuisance parameter models.

Includes doubly robust statistical inference option [10] in function ```TLP.dr_target_multigroup```

## Example Usage:


```python
import numpy as np
import pandas as pd
from TLP import TLP
import scipy
from scipy.special import logit, expit


def gen_data(N):  # following example in https://migariane.github.io/TMLE.nb.html
    w1 = np.random.binomial(1, 0.5, size=N)
    w2 = np.random.binomial(1, 0.65, size=N)
    w3 = np.round(np.random.uniform(0,4, N), 3)
    w4 = np.round(np.random.uniform(0,5, N), 3)
    ua = expit(-0.4 + 0.2* w2 + 0.15 * w3 + 0.2 * w4 + 0.15 * w2 * w4)
    A = np.random.binomial(1, ua, size=N)
    uy1 = expit(-1 + 1 - 0.1*w1 + 0.3 * w2 + 0.25 * w3 + 0.2*w4 + 0.15*w2*w4)
    uy0 = expit(-1 + 0 - 0.1*w1 + 0.3 * w2 + 0.25 * w3 + 0.2*w4 + 0.15*w2*w4)
    Y1 = np.random.binomial(1, uy1, size=N)
    Y0 = np.random.binomial(1, uy0, size=N)
    Y = Y1*A + Y0*(1-A)
    cols = ['w1', 'w2', 'w3', 'w4', 'A', 'Y', 'Y1', 'Y0']
    df = pd.DataFrame([w1, w2, w3, w4, A, Y, Y1, Y0]).T
    df.columns = cols
    
    return df

df = gen_data(100000)

true_psi = (df['Y1'] - df['Y0']).mean()
print('TRUE EFFECT', true_psi)
true_sd = scipy.stats.sem(df['Y1'] - df['Y0'])
print('TRUE SE', true_sd)

df = gen_data(1000)
sample_psi = (df['Y1'] - df['Y0']).mean()
print('SAMPLE EFFECT', sample_psi)
sample_sd = scipy.stats.sem(df['Y1'] - df['Y0'])
print('SAMPLE SE', sample_sd)
cols = df.columns

print(cols, len(cols))

# standardize the data for causal discovery
df = df.drop(['Y1', 'Y0'], 1)
cols = df.columns
print(cols, len(cols))

outcome = 'Y'
cause = 'A'
confs = ['w1', 'w2', 'w3', 'w4']
precs = []

# step 2
est_dict_G = ['LR', 'NB', 'MLP','SV', 'poly', 'RF','AB']
est_dict_Q = ['LR', 'NB', 'MLP','SV', 'poly', 'RF','AB']

# step 3
i = 0

# step 4
print('outcome: ', outcome, '. cause: ', cause, '. Confounders:', confs, '. Precisions:', precs, '\n')

# step 5
outcome_type = 'cls'  # 'reg' or 'cls'

# step 6
group_comparisons =[[1,0]]  # comparison in list format with 'group B [vs] reference_group'

# step 7
k = 5  # number of folds for SL training

# step 8
tlp = TLP(df, cause=cause, outcome=outcome, confs=list(confs),
          precs=list(precs), outcome_type=outcome_type, Q_learners=est_dict_Q, G_learners=est_dict_G,
         outcome_upper_bound=None, outcome_lower_bound=None)

# step 9 
 # fit SuperLearners
all_preds_Q, gs_Q, all_preds_G, gts_G = tlp.fit(k=k, standardized_outcome=False, calibrationQ=True, calibrationG=False)

# step 9 
# 'do' targeted learning
pre_update_effects, post_update_effects, ses, ps = tlp.target_multigroup(group_comparisons=group_comparisons)
# step 11
# 'do' doubly robust inference targeted learning [10.]
pre_update_effects_dr, post_update_effects_dr, ses_dr, ps_dr = tlp.dr_target_multigroup(group_comparisons=group_comparisons, k=5, iterations=10)

print(' REGULAR TARGETED LEARNING---------')

print('TRUE', true_psi)
print('SAMPLE', sample_psi)

print('Pre-update error:', np.abs(sample_psi - pre_update_effects['[1, 0]']))
print('Post-update error:', np.abs(sample_psi - post_update_effects['[1, 0]']))

print('TRUE SE:', true_sd)
print('SAMPLE SE:', sample_sd )
print('EST SE:', ses['[1, 0]'][0])

print('\n DOUBLE TARGETED LEARNING---------')

print('Pre-update error:', np.abs(sample_psi - pre_update_effects_dr['[1, 0]']))
print('Post-update error:', np.abs(sample_psi - post_update_effects_dr['[1, 0]']))
print('EST SE:', ses_dr['[1, 0]'][0])
```


### Outstanding Work

- Longitudinal (following G-computation/estimation) [7,8]
- Highly Adaptive Lasso [6]
- Continuous treatment/exposure 


### References
[1.] Van der Laan, M. J., & Rose, S. (2011). Targeted learning: causal inference for observational and experimental data (Vol. 4). New York: Springer

[2.] Luque‐Fernandez, M. A., Schomaker, M., Rachet, B., & Schnitzer, M. E. (2018). Targeted maximum likelihood estimation for a binary treatment: A tutorial. Statistics in medicine, 37(16), 2530-2546.

[3.] Coyle, J. R., Hejazi, N. S., Malenica, I., Phillips, R. V., Arnold, B. F., Mertens, A., ... & van der Laan, M. J. (2020). Targeting learning: robust statistics for reproducible research. arXiv preprint arXiv:2006.07333.

[4.] Van Der Laan, M. J., & Rubin, D. (2006). Targeted maximum likelihood learning. The international journal of biostatistics, 2(1).

[5.] Van der Laan, M. J., Polley, E. C., & Hubbard, A. E. (2007). Super learner. Statistical applications in genetics and molecular biology, 6(1).

[6.] Benkeser, D., & Van Der Laan, M. (2016, October). The highly adaptive lasso estimator. In 2016 IEEE international conference on data science and advanced analytics (DSAA) (pp. 689-696). IEEE.

[7.] Petersen, M., Schwab, J., Gruber, S., Blaser, N., Schomaker, M., & van der Laan, M. (2014). Targeted maximum likelihood estimation for dynamic and static longitudinal marginal structural working models. Journal of causal inference, 2(2), 147-185.

[8.] van der Laan, M. J., & Gruber, S. (2012). Targeted minimum loss based estimation of causal effects of multiple time point interventions. The international journal of biostatistics, 8(1).

[9.] Hines, O., Dukes, O., Diaz-Ordaz, K., & Vansteelandt, S. (2022). Demystifying statistical learning based on efficient influence functions. The American Statistician, 1-13.

[10.] Benkeser, D. Carone, M. van der Laan, M.J. & Gilbert, P.B. (2017) Doubly robust nonparametric inference on the average treatment effect. Biometrika 104(4) 863-880


