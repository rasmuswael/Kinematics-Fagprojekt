from inverse_kinematics.quick_setup import *

seed = 114
type = 'hmmGauss'

### Hent modelparametre her
# model_fname = "hmmGauss_experiment1"
# path = f"./models/{model_fname}"
# parameters = load_param(path, type)

### Eller træn en ny her

trainselected, testselected = get_trainandtestsample_names(seed)
print(len(trainselected), len(testselected))
print(testselected)
data_pool, sample_rate, limit = 'manual', 6, None
X, len_array, indices = quick_setup(None, data_pool=trainselected, sample_rate=sample_rate, limit=None, excluded=[])

# Example hyperparameters for hmmGauss
n_states = 25
hyperparameters = (n_states, len_array, 'full')

savepath = f"hmmGauss_{n_states}hstates"
prior = train_prior(X, indices, type, hyperparameters=hyperparameters, savepath=savepath)

# prior = get_prior(type, parameters)
#
# goal_joints = ['rfoot', 'lfoot']
#
# examples = selected
#
# np.random.seed(seed)
# samples = get_motion_samples(examples, 100, 2, sample_rate=sample_rate)
# sequences, trunc_samples = get_goal_sequences(goal_joints, samples, indices, return_trunc_samples=True)
#
# saveframes, plot = True, False
#
# n_epochs, lr, weight_decay, lh_var = 100, 1e-1 * sample_rate, 0, 1e-2
# parameters = (n_epochs, lr, weight_decay, lh_var)
#
# # inv_noprior = Inverse_model(noprior, excluded, saveframes=saveframes, plot=plot)
# # inv_normal = Inverse_model(normprior, excluded, saveframes=saveframes, plot=plot)
# # inv_gm = Inverse_model(gmprior, indices, saveframes=saveframes, plot=plot)
# inv_hmmGauss = Inverse_model(hmmGaussprior, indices, saveframes=saveframes, plot=plot)
#
#
# gen_timeseries(inv_hmmGauss, sequences, parameters, trunc_samples, view=True, init_pose='first pose', opt_pose='self',
#                interpolate=True, sample_rate=sample_rate)