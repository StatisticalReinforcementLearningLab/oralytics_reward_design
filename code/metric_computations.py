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

ENV_NAMES = ["U_LOW", "U_MED", "U_HIGH"]
ALG_VALS = range(0, 180, 10)

READ_PATH_PREFIX = "./pickle_results/"
WRITE_PATH_PREFIX = "./figures/"

def create_grid(env_name, metric_function):
    grid = np.zeros((len(ALG_VALS), len(ALG_VALS)))
    for i ALG_VALS:
        for j in ALG_VALS:
            string_prefix = "{}_{}_{}".format(i, j, env_name)
            grid[i][j], _ = metric_function(string_prefix)

    return grid

def format_rewards(string_prefix):
  total_rewards = np.zeros(shape=(NUM_TRIALS, NUM_TRIAL_USERS, NUM_DECISION_POINTS))

  # extract pickle
  for i in range(NUM_TRIALS):
    pickle_name = READ_PATH_PREFIX + string_prefix + "_{}_result.p".format(i)
    result_list = pickle.load(open(pickle_name, "rb"))
    trial_rewards = np.array([result_list[j][1]['rewards'] for j in range(len(result_list))])
    total_rewards[i] = trial_rewards

  return total_rewards

 # average across trajectories, average across users,
#  average across trials
def report_mean_reward(string_prefix):
  total_rewards = format_rewards(string_prefix)
  a = np.mean(np.mean(total_rewards, axis=2), axis=1)

  return np.mean(a), stats.sem(a)

# average across trajectories, lower 25th percentile of users,
# average across trials
def report_lower_25_reward(string_prefix):
  total_rewards = format_rewards(string_prefix)
  a = np.percentile(np.mean(total_rewards, axis=2), 25, axis=1)

  return np.mean(a), stats.sem(a)

def plot_heatmap(grid, color_scheme, file_name=None, save_fig=False):
    ax = sns.heatmap(grid, cmap=sns.color_palette(color_scheme, as_cmap=True))
    # ax.set_title(plot_title)
    if save_fig:
        fig = ax.get_figure()
        fig.savefig("{}.pdf".format(file_name))

for SIM_ENV in ENV_NAMES:
    try:
        avg_grid = create_grid(SIM_ENV, report_mean_reward)
        low_25_grid = create_grid(SIM_ENV, report_lower_25_reward)
        plot_heatmap(avg_grid, "ch:start=.2,rot=-.3", "avg_heatmap_{}".format(SIM_ENV), True)
        plot_heatmap(low_25_grid, "ch:s=-.2,r=.6", "25_perc_heatmap_{}".format(SIM_ENV), True)
    except:
        print("Couldn't for {}".format(SIM_ENV))
