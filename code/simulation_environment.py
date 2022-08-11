# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from scipy.stats import bernoulli
from scipy.stats import poisson

ROBAS_3_PARAMS_DF = pd.read_csv('../sim_env_data/non_stat_zero_infl_pois_model_params.csv')
robas_3_data_df = pd.read_csv('../sim_env_data/robas_3_data.csv')
ROBAS_3_USERS = np.array(ROBAS_3_PARAMS_DF['User'])
NUM_USERS = len(ROBAS_3_USERS)

### NORMALIZTIONS ###
def normalize_total_brush_quality(quality):
  return (quality - 154) / 163

def normalize_day_in_study(day):
  return (day - 35.5) / 34.5

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

### STATE SPACE ###
def get_user_df(user_id):
  return robas_3_data_df[robas_3_data_df['ROBAS ID'] == user_id]

def generate_state_spaces_non_stat(user_df, num_days):
  ## init ##
  D = 6
  states = np.zeros(shape=(2 * num_days, D))
  for i in range(len(states)):
    # time of day
    states[i][0] = i % 2
    # day in study
    states[i][2] = normalize_day_in_study(i // 2 + 1)
    # bias term
    states[i][5] = 1

  # reinput weekday vs. weekend
  first_weekend_idx = np.where(np.array(user_df['Day Type']) == 1)[0][0]
  for j in range(4):
    states[first_weekend_idx + j::14,4] = 1

  return states

### ENVIRONMENT AND ALGORITHM STATE SPACE FUNCTIONS ###
def get_previous_day_total_brush_quality(Qs, time_of_day, j):
    if j > 1:
        if time_of_day == 0:
            return Qs[j - 1] + Qs[j - 2]
        else:
            return Qs[j - 2] + Qs[j - 3]
    # first day there is no brushing
    else:
        return 0

def non_stat_process_env_state(session, j, Qs):
    env_state = session.copy()
    # session type - either 0 or 1
    session_type = int(env_state[0])
    # update previous day total brush time
    previous_day_total_rewards = get_previous_day_total_brush_quality(Qs, session[0], j)
    env_state[1] = normalize_total_brush_quality(previous_day_total_rewards)
    # proportion of past success brushing
    if (j >= 14):
      env_state[3] = np.sum([Qs[j - k] > 0.0 for k in range(1, 15)]) / 14

    return env_state

"""## Generate States
---
"""

NUM_DAYS = 70
# dictionary where key is index and value is user_id
USER_INDICES = {}

# dictionary where key is user id and values are lists of sessions of trial
USERS_SESSIONS_NON_STAT = {}
for i, user_id in enumerate(ROBAS_3_USERS):
  user_idx = i
  USER_INDICES[user_idx] = user_id
  user_df = get_user_df(user_id)
  USERS_SESSIONS_NON_STAT[user_id] = generate_state_spaces_non_stat(user_df, NUM_DAYS)

def get_zero_infl_params_for_user(user):
  param_dim = 6
  user_row = np.array(ROBAS_3_PARAMS_DF[ROBAS_3_PARAMS_DF['User'] == user])
  bern_params = user_row[0][2:2 + param_dim]
  poisson_params = user_row[0][2 + param_dim:]

  # poisson parameters, bernouilli parameters
  return bern_params, poisson_params

"""### Functions for Environment Models
---
"""

def linear_model(state, params):
  return state @ params

def construct_zero_infl_pois_model_and_sample(user, state, action, \
                                          bern_params, \
                                          poisson_params, \
                                          effect_func_bern=lambda state : 0, \
                                          effect_func_poisson=lambda state : 0):
  bern_linear_comp = linear_model(state, bern_params)
  if (action == 1):
    bern_linear_comp += effect_func_bern(state)
  bern_p = 1 - sigmoid(bern_linear_comp)
  # bernoulli component
  rv = bernoulli.rvs(bern_p)
  if (rv):
    # poisson component
    x = linear_model(state, poisson_params)
    if (action == 1):
      x += effect_func_poisson(state)

    l = np.exp(x)
    sample = poisson.rvs(l)

    return sample

  else:
    return 0

"""## Constructing the Effect Sizes
---
"""

bern_param_titles = ['Time.of.Day.Bern', \
                     'Prior.Day.Total.Brush.Time.norm.Bern', \
                     'Proportion.Brushed.In.Past.7.Days.Bern', \
                     'Day.Type.Bern']

poisson_param_titles = ['Time.of.Day.Poisson', \
                        'Prior.Day.Total.Brush.Time.norm.Poisson', \
                     'Proportion.Brushed.In.Past.7.Days.Poisson', \
                     'Day.Type.Poisson']

# effect size mean
zero_infl_bern_mean = -1.0 * np.mean(np.mean(np.abs(np.array([ROBAS_3_PARAMS_DF[title] for title in bern_param_titles])), axis=1))
zero_infl_poisson_mean = np.mean(np.mean(np.abs(np.array([ROBAS_3_PARAMS_DF[title] for title in poisson_param_titles])), axis=1))

# effect size std
zero_infl_bern_std = np.std(np.mean(np.abs(np.array([ROBAS_3_PARAMS_DF[title] for title in bern_param_titles])), axis=0))
zero_infl_poisson_std = np.std(np.mean(np.abs(np.array([ROBAS_3_PARAMS_DF[title] for title in poisson_param_titles])), axis=0))

## simulating the effect sizes per user ##
np.random.seed(1)
user_zero_infl_bern_effect_sizes = np.random.normal(loc=zero_infl_bern_mean, scale=zero_infl_bern_std, size=len(USERS_SESSIONS_NON_STAT.keys()))
user_zero_infl_poisson_effect_sizes = np.random.normal(loc=zero_infl_poisson_mean, scale=zero_infl_poisson_std, size=len(USERS_SESSIONS_NON_STAT.keys()))

ZERO_INFL_BERN_EFFECT_SIZE = {}
ZERO_INFL_POISSON_EFFECT_SIZES = {}

for i, user_id in enumerate(USERS_SESSIONS_NON_STAT.keys()):
  ZERO_INFL_BERN_EFFECT_SIZE[user_id] = user_zero_infl_bern_effect_sizes[i]
  ZERO_INFL_POISSON_EFFECT_SIZES[user_id] = user_zero_infl_poisson_effect_sizes[i]

ZERO_INFL_BERN_EFFECT_SIZE

ZERO_INFL_POISSON_EFFECT_SIZES

## USER-SPECIFIC EFFECT SIZES ##
# Context-Aware with all features same as baseline features excpet for Prop. Non-Zero Brushing In Past 7 Days
# which is index 3 for non stat models

non_stat_user_spec_effect_func_zero_infl_bern = lambda state, effect_size: np.array(5 * [effect_size]) @ np.delete(state, 3)
non_stat_user_spec_effect_func_zero_infl_poisson = lambda state, effect_size: np.array(5 * [effect_size]) @ np.delete(state, 3)

"""## Creating Simulation Environment Objects
---
"""

### These values are chosen by domain experts
UNRESPONSIVE_THRESHOLD = 3

class UserEnvironment():
    def __init__(self, user_id, unresponsive_val):
        # vector: size (T, D) where D = 6 is the dimension of the env. state
        # T is the length of the study
        self.user_states = USERS_SESSIONS_NON_STAT[user_id]
        # tuple: float values of effect size on bernoulli, poisson components
        self.user_effect_sizes = np.array([ZERO_INFL_BERN_EFFECT_SIZE[user_id], ZERO_INFL_POISSON_EFFECT_SIZES[user_id]])
        # float: unresponsive scaling value
        self.unresponsive_val = unresponsive_val
        # reward generating function
        self.reward_generating_func = lambda state, action: construct_zero_infl_pois_model_and_sample(user_id, state, action, \
                                          get_zero_infl_params_for_user(user_id)[0], \
                                          get_zero_infl_params_for_user(user_id)[1], \
                                          effect_func_bern=lambda state: non_stat_user_spec_effect_func_zero_infl_bern(state, self.user_effect_sizes[0]), \
                                          effect_func_poisson=lambda state: non_stat_user_spec_effect_func_zero_infl_poisson(state, self.user_effect_sizes[1]))

    def generate_reward(self, state, action):
          return self.reward_generating_func(state, action)

    def update_responsiveness(self, a1_cond, a2_cond, b_cond):
        if ((b_cond and a1_cond) or a2_cond):
            self.user_effect_sizes = self.user_effect_sizes * self.unresponsive_val

    def get_states(self):
        return self.user_states

    def get_user_effect_sizes(self):
        return self.user_effect_sizes

def create_user_envs(users_list, unresponsive_val):
    all_user_envs = {}
    for i, user in enumerate(users_list):
      new_user = UserEnvironment(user_id, unresponsive_val)
      all_user_envs[i] = new_user

    return all_user_envs

class SimulationEnvironment():
    def __init__(self, users_list, unresponsive_val):
        # Func
        self.process_env_state = non_stat_process_env_state
        # Dict: key: String user_id, val: user environment object
        self.all_user_envs = create_user_envs(users_list, unresponsive_val)
        # List: users in the environment (can repeat)
        self.users_list = users_list

    def generate_rewards(self, user_idx, state, action):
        return self.all_user_envs[user_idx].generate_reward(state, action)

    def get_states_for_user(self, user_idx):
        return self.all_user_envs[user_idx].get_states()

    def get_users(self):
        return self.users_list

    def update_responsiveness(self, user_idx, a1_cond, a2_cond, b_cond):
        self.all_user_envs[user_idx].update_responsiveness(a1_cond, a2_cond, b_cond)

### SIMULATION ENV AXIS VALUES ###
# These are the values you can tweak for the variants of the simulation environment
RESPONSIVITY_SCALING_VALS = [0, 0.25, 0.5]

LOW_R = lambda users_list: SimulationEnvironment(users_list, RESPONSIVITY_SCALING_VALS[0])
MED_R = lambda users_list: SimulationEnvironment(users_list, RESPONSIVITY_SCALING_VALS[1])
HIGH_R = lambda users_list: SimulationEnvironment(users_list, RESPONSIVITY_SCALING_VALS[2])
