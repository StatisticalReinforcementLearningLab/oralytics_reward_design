from rl_experiments import *
from absl import app
from absl import flags

# abseil ref: https://abseil.io/docs/python/guides/flags
FLAGS = flags.FLAGS
flags.DEFINE_string('sim_env_type', None, 'input the simulation environment type')
flags.DEFINE_string('algorithm_vals', None, 'input the RL algorithm candidate type')
flags.DEFINE_integer('seed', None, 'seed for np.random.seed() to reproduce the trial')

NUM_TRIAL_USERS = 72
def get_user_list(study_idxs):
    user_list = [USER_INDICES[idx] for idx in study_idxs]

    return user_list

# parses argv to access FLAGS
def main(_argv):
    # draw different users per trial
    np.random.seed(FLAGS.seed)
    print("SEED: ", FLAGS.seed)
    # denotes a weekly update schedule
    UPDATE_CADENCE = 13
    STUDY_IDXS = np.random.choice(NUM_USERS, size=NUM_TRIAL_USERS)
    print(STUDY_IDXS)

    ## HANDLING RL ALGORITHM CANDIDATE ##
    cost_params = [int(x) for x in FLAGS.algorithm_vals.split('_')]
    print("PROCESSED CANDIDATE VALS {}".format(candidate_type))
    alg_candidate = BayesianLinearRegression(cost_params, UPDATE_CADENCE)

    # get user ids corresponding to index
    USERS_LIST = get_user_list(STUDY_IDXS)
    print(USERS_LIST)

    ## HANDLING SIMULATION ENVIRONMENT ##
    env_type = FLAGS.sim_env_type
    if env_type == 'D_LOW_U_LOW':
        environment_module = D_LOW_U_LOW(USERS_LIST)
    elif env_type == 'D_LOW_U_MED':
        environment_module = D_LOW_U_MED(USERS_LIST)
    elif env_type == 'D_LOW_U_HIGH':
        environment_module = D_LOW_U_HIGH(USERS_LIST)
    elif env_type == 'D_MED_U_LOW':
        environment_module = D_MED_U_LOW(USERS_LIST)
    elif env_type == 'D_MED_U_MED':
        environment_module = D_MED_U_MED(USERS_LIST)
    elif env_type == 'D_MED_U_HIGH':
        environment_module = D_MED_U_HIGH(USERS_LIST)
    elif env_type == 'D_HIGH_U_LOW':
        environment_module = D_HIGH_U_LOW(USERS_LIST)
    elif env_type == 'D_HIGH_U_MED':
        environment_module = D_HIGH_U_MED(USERS_LIST)
    elif env_type == 'D_HIGH_U_HIGH':
        environment_module = D_HIGH_U_HIGH(USERS_LIST)
    else:
        print("ERROR: NO ENV_TYPE FOUND - ", env_type)

    print("PROCESSED ENV_TYPE {}".format(env_type))

    ## RUN EXPERIMENT ##
    # Full Pooling with Incremental Recruitment
    results = run_incremental_recruitment_exp(pre_process_users(USERS_LIST), alg_candidate, environment_module)

    pickling_location = '/n/home02/atrella/processed_pickle_results/{}_{}_{}_processed_result.p'.format(FLAGS.algorithm_candidate, FLAGS.sim_env_type, FLAGS.seed)

    ## results is a list of tuples where the first element of the tuple is user_id and the second element is a dictionary of values
    print("TRIAL DONE, PICKLING NOW")
    print("PICKLING TO: {}".format(pickling_location))
    with open(pickling_location, 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    app.run(main)
