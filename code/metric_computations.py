# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set_theme()

from simulation_environment import *
from rl_algorithm import *

### GLOBAL VALUES ###
NUM_TRIAL_USERS = 72
NUM_DECISION_POINTS = 140
NUM_TRIALS = 100
GRID_INCREMENT = 20

ENV_NAMES = ["LOW_R", "MED_R", "HIGH_R"]
ALG_VALS = range(0, 190, GRID_INCREMENT)

READ_PATH_PREFIX = "../pickle_results/"
WRITE_PATH_PREFIX = "../figures/"

def format_rewards(string_prefix):
  total_rewards = np.zeros(shape=(NUM_TRIALS, NUM_TRIAL_USERS, NUM_DECISION_POINTS))

  # extract pickle
  for i in range(NUM_TRIALS):
    try:
        pickle_name = READ_PATH_PREFIX + string_prefix + "_{}_result.p".format(i)
        result_list = pickle.load(open(pickle_name, "rb"))
        trial_rewards = np.array([result_list[j][1]['qualities'] for j in range(len(result_list))])
        total_rewards[i] = trial_rewards
    except:
        print("Couldn't for {}".format(pickle_name))


  return total_rewards

 # average across trajectories, average across users,
#  average across trials
def report_mean_reward(total_rewards):
  a = np.mean(np.mean(total_rewards, axis=2), axis=1)

  return np.mean(a), stats.sem(a)

# average across trajectories, lower 25th percentile of users,
# average across trials
def report_lower_25_reward(total_rewards):
  a = np.percentile(np.mean(total_rewards, axis=2), 25, axis=1)

  return np.mean(a), stats.sem(a)

def report_upper_75_reward(total_rewards):
  a = np.percentile(np.mean(total_rewards, axis=2), 75, axis=1)

  return np.mean(a), stats.sem(a)

def create_grids(env_name):
    avg_grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    low_perc_grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    high_perc_grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    for i in ALG_VALS:
        for j in ALG_VALS:
            string_prefix = "{}_{}_{}".format(env_name, i, j)
            print("For: ", string_prefix)
            total_rewards = format_rewards(string_prefix)
            avg_grid[int(i / GRID_INCREMENT)][int(j / GRID_INCREMENT)], _ = report_mean_reward(total_rewards)
            low_perc_grid[int(i / GRID_INCREMENT)][int(j / GRID_INCREMENT)], _ = report_lower_25_reward(total_rewards)
            high_perc_grid[int(i / GRID_INCREMENT)][int(j / GRID_INCREMENT)], _ = report_upper_75_reward(total_rewards)


    return avg_grid, low_perc_grid, high_perc_grid

for SIM_ENV in ENV_NAMES:
    avg_grid, low_25_grid, high_75_grid = create_grids(SIM_ENV)
    with open(WRITE_PATH_PREFIX + "{}_AVG_HEATMAP.p".format(SIM_ENV), 'wb') as f:
        pickle.dump(avg_grid, f)
    with open(WRITE_PATH_PREFIX + "{}_25_PERC_HEATMAP.p".format(SIM_ENV), 'wb') as f:
        pickle.dump(low_25_grid, f)
    with open(WRITE_PATH_PREFIX + "{}_75_PERC_HEATMAP.p".format(SIM_ENV), 'wb') as f:
        pickle.dump(high_75_grid, f)
