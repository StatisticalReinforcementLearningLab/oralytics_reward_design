# -*- coding: utf-8 -*-
"""
# Forming the Prior for the Oralytics RL Algorithm Using ROBAS 2 Data
---
For the Oralytics RL Algorithm, we have the following baseline feature space:

1. Time of day
2. Prior total brushing duration (in the same brushing window during the past 24 hours prior to decision time)
3. Weekend vs. weekday
4. Intercept term

## Pulling ROBAS 2 Dataset
---
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

import pymc3 as pm
from pymc3.model import Model
import theano.tensor as tt
import arviz as az

df = pd.read_csv('../sim_env_data/robas_2_data.csv')
alg_features_df = df[['User', 'Time of Day', 'Brush Time', 'Day Type']]
# num. of baseline featuers
D = 4

alg_features_df

# normalized values derived from ROBAS 2 dataset
def normalize_total_brush_time(time):
  return (time - 172) / 118

# grab user specific df
def get_user_df(user_id):
  return alg_features_df[alg_features_df['User'] == user_id]

def generate_state_spaces_for_single_user(user_id, rewards):
  ## init ##
  user_df = get_user_df(user_id)
  states = np.zeros(shape=(len(user_df), D))

  for i in range(len(user_df)):
    df_array = np.array(user_df)[i]
    # time of day
    states[i][0] = df_array[1]
    # prior day brushing quality
    if i > 1:
      if states[i][0] == 0:
        states[i][1] = normalize_total_brush_time(rewards[i - 1] + rewards[i - 2])
      else:
        states[i][1] = normalize_total_brush_time(rewards[i - 2] + rewards[i - 3])
    # weekday or weekend term
    states[i][2] = df_array[3]
    # bias term
    states[i][3] = 1

  return states

## making a dictionary of user, trajectory (states, rewards)
num_users = len(alg_features_df['User'].unique())
num_days = 28
num_sessions = 2 * num_days

# dictionary where key is user id and values are lists of sessions of trial
users_sessions = {}
total_X = np.empty(shape=(1, D))
total_Y = np.empty(shape=1)
for user_id in alg_features_df['User'].unique():
  rewards = np.array(alg_features_df.loc[alg_features_df['User'] == user_id]['Brush Time'])
  states = generate_state_spaces_for_single_user(user_id, rewards)
  total_X = np.concatenate((total_X, states), axis=0)
  total_Y = np.concatenate((total_Y, rewards), axis=None)
  users_sessions[user_id] = [states, rewards]

total_X = total_X[1:,]
total_Y = total_Y[1:]

"""## Fitting $\sigma_n^2$
---
1. We fit one linear regression model per user.
2. We then obtain the weights for each fitted model and calculate residuals.
3. $\sigma_n$ is set to the average SD of the residuals.

Closed Form soluation for linear regression:
$w^* = (X^TX)^{-1}X^Ty$
"""

# fit one sigma_n per user to find the variance of sigma_n
sigma_n_squared_s = []

for user in users_sessions.keys():
  states, rewards = users_sessions[user]
  user_w = np.linalg.inv(states.T @ states) @ states.T @ rewards

  user_predicted_Y =  states @ user_w
  user_residuals = rewards - user_predicted_Y

  sigma_n_squared_s.append(np.var(user_residuals))

print("sigma_squared", np.mean(sigma_n_squared_s))

"""## Fitting $\mu_{\alpha_0}, \Sigma_{\alpha_0}$
---
We fit our prior parameters in accordance to the procedure in [[Liao et. al., 2015]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8439432/#R23) Section 6.3.
1. We use GEE regression anaylsis to fit a model per user to identify significance of each feature.
2. For features that are significant, we set prior mean (mean across users) to be the point estimate from this analysis. Also, prior SD is the empirical SD across participant models (across users) from participant-specific GEE analysis.

  For non-significant features, we set the prior mean to be 0 and the SD is shrunk in half.

$\Sigma_0$ is a diagonal matrix whose diagonal entries are the correspnoding variances for each feature.

Variance estimator for $w^*$:

$(X^\top X)^{-1} (\sum_{i=1}^n \left\{ \sum_{t=1}^T X_{i,t} r_{i,t} \right\}^{\otimes 2} ) (X^\top X)^{-1}$

Note: $(\sum_{i=1}^n \left\{ \sum_{t=1}^T X_{i,t} r_{i,t} \right\}^{\otimes 2} ) \in R^{dxd}$, $\otimes 2$ denotes outer product

### 1. Significance Test For Each Feature
---
"""

def calculate_var_estimator(X):
  matrix = np.zeros(shape=(4, 4))
  for user in users_sessions.keys():
    user_state = users_sessions[user][0]
    user_rewards = users_sessions[user][1]
    vector = np.array([user_state[i] * user_rewards[i] for i in range(len(user_state))])
    matrix += vector.T @ vector

  return np.linalg.inv(X.T @ X) @ matrix @ np.linalg.inv(X.T @ X)

var_matrix = calculate_var_estimator(total_X)

w = np.linalg.inv(total_X.T @ total_X) @ total_X.T @ total_Y

[abs(w[i] / var_matrix[i][i]**(0.5)) for i in range(len(w))]

"""This is the cut off value. If the test statistic (calculated above) is greater than the cut off value, then the feature is significiant.

Cut Off Value = $|\text{inverse CDF} (\text{significance} / 2, \text{num. of users})|$
"""

print("Significance Cut Off Value:", abs(scipy.stats.t.ppf(0.05 / 2, 32)))

"""### Step 2. """

ws = []
for user in users_sessions.keys():
  states, rewards = users_sessions[user]
  user_w = np.linalg.inv(states.T @ states) @ states.T @ rewards
  ws.append(user_w)

# std of ws fitted across users
sds = np.std(ws, axis=0)
### diagonal of Sigma_{\alpha_0} ###
print("diag of Sigma_{alpha_0}: ", [sds[0] / 2, sds[1], sds[2] / 2, sds[3]])

mus = np.mean(ws, axis=0)
### mu_{\alpha_0} ###
print("mu_{alpha}: ",[0, mus[1], 0, mus[3]])

### sigma_{\beta} ###
print("sigma_{beta}: ", np.mean([sds[0] / 2, sds[1], sds[2] / 2, sds[3]]))
