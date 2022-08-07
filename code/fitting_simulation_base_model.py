# -*- coding: utf-8 -*-
"""
Fits the environment base model for each user
using the ROBAS 3 data set
"""

import pandas as pd
import numpy as np
import pymc3 as pm
from pymc3.model import Model
import theano.tensor as tt
import arviz as az

ROBAS_3_DATA = pd.read_csv("robas_3_data.csv")

ROBAS_3_USERS = np.unique(ROBAS_3_DATA['ROBAS ID'])
NUM_USERS = len(ROBAS_3_USERS)
print("ROBAS 3 Phase 2 has {} users.".format(NUM_USERS))
print(ROBAS_3_USERS)

"""## Formatting States and Rewards
---
"""

# total brushing quality
robas_3_user_total_brush_quality = (np.array(ROBAS_3_DATA['Brushing duration'])[::2] - np.array(ROBAS_3_DATA['Pressure duration'])[::2])\
 + (np.array(ROBAS_3_DATA['Brushing duration'])[1::2] - np.array(ROBAS_3_DATA['Pressure duration'])[1::2])

print("Empirical Mean: ", np.mean(robas_3_user_total_brush_quality))
print("Empirical Std: ", np.std(robas_3_user_total_brush_quality))

# Z-score normalization
def normalize_total_brush_quality(quality):
  return (quality - np.mean(robas_3_user_total_brush_quality)) / np.std(robas_3_user_total_brush_quality)

# returns a function to normalize day in study for each user
def normalize_day_in_study_func(user_id):
  user_specific_length = np.array(ROBAS_3_DATA[ROBAS_3_DATA['ROBAS ID'] == user_id]['Day in Study'])[-1]

  return lambda day: (day - ((user_specific_length + 1)/2)) / ((user_specific_length - 1)/2)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def get_rewards(user_id):
  return np.array(ROBAS_3_DATA[ROBAS_3_DATA['ROBAS ID'] == user_id]['Brushing duration'] - \
                  ROBAS_3_DATA[ROBAS_3_DATA['ROBAS ID'] == user_id]['Pressure duration'])

def get_user_df(user_id):
  return ROBAS_3_DATA[ROBAS_3_DATA['ROBAS ID'] == user_id]

# generating non-stationary state space
# 0 - Time of Day
# 1 - Prior Day Total Brush Time
# 2 - Day In Study
# 3 - Prop. Non-Zero Brushing In Past 7 Days
# 4 - Weekday vs. Weekend
# 5 - Bias

def generate_state_spaces_non_stationarity(user_id, rewards):
  ## init ##
  D = 6
  user_df = get_user_df(user_id)
  states = np.zeros(shape=(len(user_df), D))
  # user specific normalization for day in study
  norm_func = normalize_day_in_study_func(user_id)
  for i in range(len(user_df)):
    df_array = np.array(user_df)[i]
    # time of day
    states[i][0] = df_array[3]
    # prior day brushing quality
    if i > 1:
      if states[i][0] == 0:
        states[i][1] = normalize_total_brush_quality(rewards[i - 1] + rewards[i - 2])
      else:
        states[i][1] = normalize_total_brush_quality(rewards[i - 2] + rewards[i - 3])
    # day in study
    states[i][2] = norm_func(df_array[2])
    # prop. brushed in past 7 days
    if i > 13:
      states[i][3] = df_array[7]
    # weekday or weekend term
    states[i][4] = df_array[6]
    # bias term
    states[i][5] = 1

  return states

# dictionary where key is user id and values are lists of sessions of trial
users_sessions_non_stationarity = {}
users_rewards = {}
for user_id in ROBAS_3_USERS:
  user_rewards = get_rewards(user_id)
  users_rewards[user_id] = user_rewards
  users_sessions_non_stationarity[user_id] = generate_state_spaces_non_stationarity(user_id, user_rewards)

"""## Fitting Zero-Inflated Poisson Model
---
"""

# ref: https://pymc3-testing.readthedocs.io/en/rtd-docs/api/distributions/discrete.html#pymc3.distributions.discrete.ZeroInflatedPoisson
# e.g. https://discourse.pymc.io/t/zero-inflated-poisson-example/3862
# https://bwengals.github.io/gps-with-non-normal-likelihoods-in-pymc3.html
def build_0_inflated_linear_model(X, Y):
  model = pm.Model()
  with Model() as model:
    d = X.shape[1]
    w_b = pm.MvNormal('w_b', mu=np.zeros(d, ), cov=np.eye(d), shape=d)
    w_p = pm.MvNormal('w_p', mu=np.zeros(d, ), cov=np.eye(d), shape=d)
    bern_term = X @ w_b
    poisson_term = X @ w_p
    R = pm.ZeroInflatedPoisson("likelihood", psi=1 - sigmoid(bern_term), theta=tt.exp(poisson_term), observed=Y)

  return model

robas4_states = users_sessions_non_stationarity['robas+4']
robas4_states_rewards = users_rewards['robas+4']
test_model = build_0_inflated_linear_model(robas4_states, robas4_states_rewards)

with test_model:
  test_map = pm.find_MAP(start={'w_b': 0.01 * np.ones(6), 'w_p': 0.01 * np.ones(6)})

test_map

pm.model_to_graphviz(test_model)

def run_zero_infl_map_for_users(users_sessions, users_rewards, d, num_restarts):
  model_params = {}

  for user_id in users_sessions.keys():
    print("FOR USER: ", user_id)
    user_states = users_sessions[user_id]
    user_rewards = users_rewards[user_id]
    logp_vals = np.empty(shape=(num_restarts,))
    param_vals = np.empty(shape=(num_restarts, 2 * d))
    for seed in range(num_restarts):
      model = build_0_inflated_linear_model(user_states, user_rewards)
      np.random.seed(seed)
      init_params = {'w_b': np.random.randn(d), 'w_p':  np.random.randn(d)}
      with model:
        map_estimate = pm.find_MAP(start=init_params)
      w_b = map_estimate['w_b']
      w_p = map_estimate['w_p']
      logp_vals[seed] = model.logp(map_estimate)
      param_vals[seed] = np.concatenate((w_b, w_p), axis=None)
    model_params[user_id] = param_vals[np.argmax(logp_vals)]

  return model_params

# non-stationary model, log normal params
non_stat_model_params = run_zero_infl_map_for_users(users_sessions_non_stationarity, users_rewards, d=6, num_restarts=5)

non_stat_zero_infl_model_columns = ['User', 'Time.of.Day.Bern', 'Prior.Day.Total.Brush.Time.norm.Bern', 'Day.in.Study.norm.Bern', 'Proportion.Brushed.In.Past.7.Days.Bern', 'Day.Type.Bern', 'Intercept.Bern', \
                        'Time.of.Day.Poisson', 'Prior.Day.Total.Brush.Time.norm.Poisson', 'Day.in.Study.norm.Poisson', 'Proportion.Brushed.In.Past.7.Days.Poisson', 'Day.Type.Poisson', 'Intercept.Poisson']

# creating the data frame
non_stationarity_df = pd.DataFrame(columns = non_stat_zero_infl_model_columns)
for user in non_stat_model_params.keys():
  values = non_stat_model_params[user]
  new_row = {}
  new_row['User'] = user
  for i in range(1, len(non_stat_zero_infl_model_columns)):
    new_row[non_stat_zero_infl_model_columns[i]] = values[i - 1]
  non_stationarity_df = non_stationarity_df.append(new_row, ignore_index=True)

non_stationarity_df.to_csv('non_stat_zero_infl_pois_model_params.csv')
