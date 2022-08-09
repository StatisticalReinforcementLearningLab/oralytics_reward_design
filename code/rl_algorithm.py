# -*- coding: utf-8 -*-
"""
RL Algorithm that uses a contextual bandit framework with Thompson sampling, full-pooling, and
a Bayesian Linear Regression reward approximating function.
"""

import pandas as pd
import numpy as np
from scipy.stats import bernoulli

## CLIPPING VALUES ##
MIN_CLIP_VALUE = 0.35
MAX_CLIP_VALUE = 0.75
# Advantage Time Feature Dimensions
D_advantage = 4
# Baseline Time Feature Dimensions
D_baseline = 5
# Number of Posterior Draws
NUM_POSTERIOR_SAMPLES = 5000

### Reward Definition ###
GAMMA = 13/14
B = 110
A_1 = 0.5
A_2 = 0.8
DISCOUNTED_GAMMA_ARRAY = GAMMA ** np.flip(np.arange(14))
CONSTANT = (1 - GAMMA) / (1 - GAMMA**14)

# brushing duration is of length 14 where the first element is the brushing duration
# at time t - 14 and the last element the brushing duration at time t - 1
def calculate_b_bar(brushing_durations):
  sum_term = DISCOUNTED_GAMMA_ARRAY * brushing_durations

  return CONSTANT * np.sum(sum_term)

def calculate_a_bar(past_actions):
  sum_term = DISCOUNTED_GAMMA_ARRAY * past_actions

  return CONSTANT * np.sum(sum_term)

def calculate_b_condition(brushing_durations):
  return calculate_b_bar(brushing_durations) > B

def calculate_a1_condition(past_actions):
  return calculate_a_bar(past_actions) > A_1

def calculate_a2_condition(past_actions):
  return calculate_a_bar(past_actions) > A_2

def cost_definition(xi_1, xi_2, action, B_condition, A1_condition, A2_condition):
  return action * (xi_1 * B_condition * A1_condition + xi_2 * A2_condition)

# returns the reward where the cost term is parameterized by xi_1, xi_2
def reward_definition(brushing_duration, pressure_duration, xi_1, xi_2, current_action,\
                      brushing_durations, past_actions):
  B_condition = calculate_b_condition(brushing_durations)
  A1_condition = calculate_a1_condition(past_actions)
  A2_condition = calculate_a2_condition(past_actions)
  Q = min(brushing_duration, 180) - pressure_duration
  C = cost_definition(xi_1, xi_2, current_action, B_condition, A1_condition, A2_condition)

  return Q - C

## HELPERS ##
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

## baseline: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - weekend vs. week day
# 4 - bias
## advantage: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - bias
def process_alg_state(env_state, past_brushing, past_actions):
    baseline_state = np.array([env_state[0], calculate_b_bar(past_brushing), \
                               calculate_a_bar(past_actions), env_state[4], 1])
    advantage_state = np.delete(baseline_state, 3)

    return advantage_state, baseline_state

GAMMA = 13/14
np.sum(((1 - GAMMA) / (1 - GAMMA**14)) * (GAMMA ** np.flip(np.arange(14))) * np.ones(14))

class RLAlgorithmCandidate():
    def __init__(self, cost_params, update_cadence):
        self.update_cadence = update_cadence
        # xi_1, xi_2 params for the cost term parameterizes the reward def. func.
        self.reward_def_func = lambda brushing_duration, pressure_duration, current_action, brushing_durations, past_actions: \
                      reward_definition(brushing_duration, pressure_duration, \
                                        cost_params[0], cost_params[1], \
                                        current_action, brushing_durations, past_actions)
        # process_alg_state is a global function
        self.process_alg_state_func = process_alg_state

    def action_selection(self, advantage_state, baseline_state):
        return 0

    def update(self, advantage_states, baseline_states, actions, pis, rewards):
        return 0

    def get_update_cadence(self):
        return self.update_cadence

"""### Bayesian Linear Regression Thompson Sampler
---

### Helper Functions
---
"""

## POSTERIOR HELPERS ##
# create the feature vector given state, action, and action selection probability
def create_big_phi(advantage_states, baseline_states, actions, probs):
  big_phi = np.hstack((baseline_states, np.multiply(advantage_states.T, probs).T, \
                       np.multiply(advantage_states.T, (actions - probs)).T,))
  return big_phi

def compute_posterior_var(Phi, sigma_n_squared, prior_sigma):
  return np.linalg.inv(1/sigma_n_squared * Phi.T @ Phi + np.linalg.inv(prior_sigma))

def compute_posterior_mean(Phi, R, sigma_n_squared, prior_mu, prior_sigma):
  # return np.linalg.inv(1/sigma_n_squared * X.T @ X + np.linalg.inv(prior_sigma)) \
  # @ (1/sigma_n_squared * X.T @ y + (prior_mu @ np.linalg.inv(prior_sigma)).T)
  return compute_posterior_var(Phi, sigma_n_squared, prior_sigma) \
   @ (1/sigma_n_squared * Phi.T @ R + np.linalg.inv(prior_sigma) @ prior_mu)

# update posterior distribution
def update_posterior_w(Phi, R, sigma_n_squared, prior_mu, prior_sigma):
  mean = compute_posterior_mean(Phi, R, sigma_n_squared, prior_mu, prior_sigma)
  var = compute_posterior_var(Phi, sigma_n_squared, prior_sigma)

  return mean, var

def get_beta_posterior_draws(posterior_mean, posterior_var):
  # grab last D_advantage of mean vector
  beta_post_mean = posterior_mean[-D_advantage:]
  # grab right bottom corner D_advantage x D_advantage submatrix
  beta_post_var = posterior_var[-D_advantage:,-D_advantage:]

  return np.random.multivariate_normal(beta_post_mean, beta_post_var, NUM_POSTERIOR_SAMPLES)

## ACTION SELECTION ##
# we calculate the posterior probability of P(R_1 > R_0) clipped
# we make a Bernoulli draw with prob. P(R_1 > R_0) of the action
def bayes_lr_action_selector(beta_posterior_draws, advantage_state):
  num_positive_preds = len(np.where(beta_posterior_draws @ advantage_state > 0)[0])
  posterior_prob =  num_positive_preds / len(beta_posterior_draws)
  clipped_prob = max(min(MAX_CLIP_VALUE, posterior_prob), MIN_CLIP_VALUE)
  return bernoulli.rvs(clipped_prob), clipped_prob

"""### BLR Algorithm Object
---
"""

## baseline: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - weekend vs. week day
# 4 - bias
## advantage: ##
# 0 - time of day
# 1 - b bar
# 2 - a bar
# 3 - bias

class BayesianLinearRegression(RLAlgorithmCandidate):
    def __init__(self, cost_params, update_cadence):
        super(BayesianLinearRegression, self).__init__(cost_params, update_cadence)

        # THESE VALUES WERE SET WITH ROBAS 2 DATA
        # size of mu vector = D_baseline + D_advantage + D_advantage
        self.PRIOR_MU = np.array([0, 4.925, 0, 0, 82.209, 0, 0, 0, 0, 0, 0, 0, 0])
        # self.PRIOR_MU = np.zeros(D_baseline + D_advantage + D_advantage)
        # self.PRIOR_SIGMA = 5 * np.eye(len(self.PRIOR_MU))
        sigma_beta = 29.624
        self.PRIOR_SIGMA = np.diag(np.array([29.090**2, 30.186**2, sigma_beta**2, 12.989**2, 46.240**2, \
                                             sigma_beta**2, sigma_beta**2, sigma_beta**2, sigma_beta**2,\
                                             sigma_beta**2, sigma_beta**2, sigma_beta**2, sigma_beta**2]))
        self.SIGMA_N_2 = 3396.449
        # initial draws are from the prior
        self.beta_posterior_draws = get_beta_posterior_draws(self.PRIOR_MU, self.PRIOR_SIGMA)

    def action_selection(self, advantage_state, baseline_state):
        return bayes_lr_action_selector(self.beta_posterior_draws, advantage_state)

    def update(self, advantage_states, baseline_states, actions, pis, rewards):
        Phi = create_big_phi(advantage_states, baseline_states, actions, pis)
        posterior_mean, posterior_var = update_posterior_w(Phi, rewards, self.SIGMA_N_2, self.PRIOR_MU, self.PRIOR_SIGMA)
        self.beta_posterior_draws = get_beta_posterior_draws(posterior_mean, posterior_var)
