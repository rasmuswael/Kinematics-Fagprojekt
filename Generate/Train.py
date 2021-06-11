from inverse_kinematics.quick_setup import *

seed = 4256

type = 'normal'
trainselected, testselected = get_trainandtestsample_names(seed)
print(len(trainselected), len(testselected))
print(testselected)
data_pool, sample_rate, limit = 'manual', 6, None
X, len_array, indices = quick_setup(None, data_pool=trainselected, sample_rate=sample_rate, limit=None, excluded=[])

print(len(indices))
# Example hyperparameters for hmmGauss
# n_states = 16
# hyperparameters = (n_states, len_array, 'full')
# savepath = f"{type}_{n_states}hstates_sr{sample_rate}"
# parameters = train_prior(X, indices, type, hyperparameters=hyperparameters, savepath=savepath)

hyperparameters = ()
savepath = f"{type}_sr{sample_rate}"
parameters = train_prior(X, indices, type, hyperparameters=hyperparameters, savepath=savepath)


