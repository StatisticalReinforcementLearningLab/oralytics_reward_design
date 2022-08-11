from rl_experiments import *
from absl import app
from absl import flags
import pickle

# abseil ref: https://abseil.io/docs/python/guides/flags
FLAGS = flags.FLAGS
flags.DEFINE_string('sim_env_type', None, 'input the simulation environment type')
flags.DEFINE_string('algorithm_val_1', None, 'input the RL algorithm candidate value for var 1')
flags.DEFINE_string('algorithm_val_2', None, 'input the RL algorithm candidate value for var 2')

NUM_TRIAL_USERS = 72
def get_user_list(study_idxs):
    user_list = [USER_INDICES[idx] for idx in study_idxs]

    return user_list

MAX_SEED_VAL = 100

# parses argv to access FLAGS
def main(_argv):
    # denotes a weekly update schedule
    UPDATE_CADENCE = 13

    ## HANDLING RL ALGORITHM CANDIDATE ##
    cost_params = [int(FLAGS.algorithm_val_1), int(FLAGS.algorithm_val_2)]
    print("PROCESSED CANDIDATE VALS {}".format(cost_params))
    alg_candidate = BayesianLinearRegression(cost_params, UPDATE_CADENCE)

    for current_seed in range(MAX_SEED_VAL):
        # draw different users per trial
        np.random.seed(current_seed)
        print("SEED: ", current_seed)

        STUDY_IDXS = np.random.choice(NUM_USERS, size=NUM_TRIAL_USERS)
        print(STUDY_IDXS)

        # get user ids corresponding to index
        USERS_LIST = get_user_list(STUDY_IDXS)
        print(USERS_LIST)

        ## HANDLING SIMULATION ENVIRONMENT ##
        env_type = FLAGS.sim_env_type
        if env_type == 'LOW_R':
            environment_module = LOW_R(USERS_LIST)
        elif env_type == 'MED_R':
            environment_module = MED_R(USERS_LIST)
        elif env_type == 'HIGH_R':
            environment_module = HIGH_R(USERS_LIST)
        else:
            print("ERROR: NO ENV_TYPE FOUND - ", env_type)

        print("PROCESSED ENV_TYPE {}".format(env_type))

        ## RUN EXPERIMENT ##
        # Full Pooling with Incremental Recruitment
        results = run_incremental_recruitment_exp(pre_process_users(USERS_LIST), alg_candidate, environment_module)

        pickling_location = '../pickle_results/{}_{}_{}_{}_result.p'.format(FLAGS.sim_env_type, cost_params[0], cost_params[1], current_seed)

        ## results is a list of tuples where the first element of the tuple is user_id and the second element is a dictionary of values
        print("TRIAL DONE, PICKLING NOW")
        print("PICKLING TO: {}".format(pickling_location))
        with open(pickling_location, 'wb') as f:
            pickle.dump(results, f)


if __name__ == '__main__':
    app.run(main)
