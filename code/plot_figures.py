# -*- coding: utf-8 -*-

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import pickle

### GLOBAL VALUES ###
ENV_NAMES = ["LOW_R", "MED_R", "HIGH_R"]
E_MAPPING = {"LOW_R": 0, "MED_R": 0.25, "HIGH_R":0.5}

READ_PATH_PREFIX = "../figures/"
WRITE_PATH_PREFIX = "../figures/"

# sns colors: https://seaborn.pydata.org/tutorial/color_palettes.html
def plot_heatmap(grid, color_scheme, title_val, file_name=None, save_fig=False):
    # clear subplots
    plt.figure()
    ax = sns.heatmap(grid,cmap=sns.color_palette(color_scheme, as_cmap=True), \
                     xticklabels=1, yticklabels=1)
    ax.invert_yaxis()
    ax.set_xlabel(r'$\xi_1$')
    ax.set_ylabel(r'$\xi_2$')
    ax.set_xticklabels(np.arange(0, 181, 20))
    ax.set_yticklabels(np.arange(0, 181, 20))
    ax.set_title(r'$E={}$'.format(title_val))

    if save_fig:
        fig = ax.get_figure()
        fig.savefig(WRITE_PATH_PREFIX + "{}.pdf".format(file_name))

for SIM_ENV in ENV_NAMES:
    print("For Env: ", SIM_ENV)
    with open(READ_PATH_PREFIX + '{}_AVG_HEATMAP.p'.format(SIM_ENV), 'rb') as f:
        avg_grid = pickle.load(f)

    with open(READ_PATH_PREFIX + '{}_25_PERC_HEATMAP.p'.format(SIM_ENV), 'rb') as f:
        low_25_grid = pickle.load(f)

    title_val = E_MAPPING[SIM_ENV]

    plot_heatmap(avg_grid.T, "ch:start=.2,rot=-.3", title_val, "avg_heatmap_{}".format(SIM_ENV), True)
    plot_heatmap(low_25_grid.T, "ch:s=-.2,r=.6", title_val, "25_perc_heatmap_{}".format(SIM_ENV), True)
