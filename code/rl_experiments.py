from simulation_environment import *
from rl_algorithm import *

## GLOBAL VALUES ##
RECRUITMENT_RATE = 4
TRIAL_LENGTH_IN_WEEKS = 10

# returns a int(NUM_USERS / RECRUITMENT_RATE) x RECRUITMENT_RATE array of user indices
# row index represents the week that they enter the study
def pre_process_users(total_trial_users):
    results = []
    for j, user in enumerate(total_trial_users):
        results.append((j, int(j // RECRUITMENT_RATE), user))

    return np.array(results)

### data structure can be a list of tuples (user_id, {rewards, actions, probs, states}) so it's easier to process
# and calculate regret for
### runs experiment with full pooling and incremental recruitment
def run_incremental_recruitment_exp(user_groups, alg_candidate, sim_env):
    # users_groups will be a list of tuples where tuple[0] is the week of entry
    # and tuple[1] is an array of user_ids

    env_users = sim_env.get_users()
    update_cadence = alg_candidate.get_update_cadence()
    init_dict = {"baseline_states": np.empty((0, D_baseline), float), "advantage_states": np.empty((0, D_advantage), float), \
             "env_states": np.empty((0, 6), float), "actions": np.empty((0, 1), float), "probs": np.empty((0, 1), float), \
             "qualities": np.empty((0, 1), float), "costs": np.empty((0, 1), float), "rewards": np.empty((0, 1), float)}
    result = [(user_id, init_dict.copy()) for user_id in env_users]
    total_results = init_dict.copy()
    current_groups = user_groups[:4]
    week = 0
    while (len(current_groups) > 0):
        print("Week: ", week)
        for user_tuple in current_groups:
            user_idx, user_entry_date = int(user_tuple[0]), int(user_tuple[1])
            user_states = sim_env.get_states_for_user(user_idx)
            # do action selection for 14 decision times (7 days)
            for decision_idx in range(14):
                ## PROCESS STATE ##
                j = (week - user_entry_date) * 14 + decision_idx
                session = user_states[j]
                env_state = sim_env.process_env_state(session, j, result[user_idx][1]["qualities"])
                # if first week we impute A bar and B bar
                if j < 14:
                    b_bar = np.mean(result[user_idx][1]["qualities"]) if len(result[user_idx][1]["qualities"]) > 0 else 0
                    a_bar = np.mean(result[user_idx][1]["actions"]) if len(result[user_idx][1]["actions"]) > 0 else 0
                else:
                    b_bar = calculate_b_bar(result[user_idx][1]["qualities"][j - 14:j])
                    a_bar = calculate_a_bar(result[user_idx][1]["actions"][j - 14:j])
                advantage_state, baseline_state = alg_candidate.process_alg_state_func(env_state, b_bar, a_bar)
                ## SAVE STATE VALUES ##
                result[user_idx][1]["env_states"] = \
                np.append(result[user_idx][1]["env_states"], env_state.reshape(1, -1), axis=0)
                result[user_idx][1]["advantage_states"] = \
                np.append(result[user_idx][1]["advantage_states"], advantage_state.reshape(1, -1), axis=0)
                total_results["advantage_states"] = np.append(total_results["advantage_states"], advantage_state.reshape(1, -1), axis=0)
                result[user_idx][1]["baseline_states"] = \
                np.append(result[user_idx][1]["baseline_states"], baseline_state.reshape(1, -1), axis=0)
                total_results["baseline_states"] = np.append(total_results["baseline_states"], baseline_state.reshape(1, -1), axis=0)
                ## ACTION SELECTION ##
                action, action_prob = alg_candidate.action_selection(advantage_state, baseline_state)
                ## SAVE ACTION VALUES ##
                result[user_idx][1]["actions"] = \
                np.append(result[user_idx][1]["actions"], action)
                total_results["actions"] = np.append(total_results["actions"], action)
                result[user_idx][1]["probs"] = \
                np.append(result[user_idx][1]["probs"], action_prob)
                total_results["probs"] = np.append(total_results["probs"], action_prob)
                ## REWARD GENERATION ##
                quality = sim_env.generate_rewards(user_idx, env_state, action)
                reward = alg_candidate.reward_def_func(quality, action, b_bar, a_bar)
                ## SAVE REWARD VALUES ##
                result[user_idx][1]["qualities"] = \
                np.append(result[user_idx][1]["qualities"], min(quality, 180))
                result[user_idx][1]["rewards"] = \
                np.append(result[user_idx][1]["rewards"], reward)
                result[user_idx][1]["costs"] = \
                np.append(result[user_idx][1]["costs"], min(quality, 180) - reward)
                total_results["rewards"] = np.append(total_results["rewards"], reward)
                ## UPDATE UNRESPONSIVENESS ##
                # only after first week
                if j >= 14:
                    sim_env.update_responsiveness(user_idx, calculate_a1_condition(a_bar),\
                     calculate_a2_condition(a_bar), calculate_b_condition(b_bar))

        # update time at the end of each week
        print("UPDATE TIME.")
        alg_candidate.update(total_results["advantage_states"], total_results["baseline_states"], \
                            total_results["actions"], total_results["probs"], total_results["rewards"])
        # handle adding or removing user groups
        week += 1
        if (week < len(user_groups)):
            # add more users
            current_groups = np.concatenate((current_groups, user_groups[RECRUITMENT_RATE * week: RECRUITMENT_RATE * week + RECRUITMENT_RATE]), axis=0)
        # check if some user group finished the study
        if (week > TRIAL_LENGTH_IN_WEEKS - 1):
            current_groups = current_groups[RECRUITMENT_RATE:]

    return result
