from InverseKinematics import *
from TimeSeries import *
seed = 1510
torch.manual_seed(seed)

sample_rate=6

selected = get_fnames(["run", "walk"])
data = parse_selected(selected, sample_rate=sample_rate, limit=20000)
len_array = get_lengths_np(data)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1] - 3)]

dummy_joints, dummy_pose = dummy()

# excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes']
excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist', 'lwrist', 'rclavicle', 'lclavicle']

included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
# mean, cov = compute_parameters_normal(truncate(X))
# mean, cov = mean[indices], cov[indices, :][:, indices]

# means, covs, weights = compute_gm_params(X, n_components=5, indices=indices, covariance_type='full')
n_states=9
model = compute_hmmGauss(X, len_array, n_components=n_states, indices=indices, covariance_type='full')


# noprior = ('noprior', None)
# normprior = ('normal', normal_prior(mean, cov))
# gmprior = ('gaussianmixture', gmm_prior(means, covs, weights))
hmmGaussprior = ('hmmGauss', (gmm_prior(model.means_, model.covars_, torch.zeros(n_states)), model))

goal_joints = ['rfoot','lfoot']

np.random.seed(seed)
samples = get_motion_samples(selected, 40, 2, sample_rate=sample_rate)
sequences = get_goal_sequences(goal_joints, samples, indices)

saveframes, plot = True, False

n_epochs, lr, weight_decay, lh_var = 100, 1e-1 * sample_rate, 0, 1e-2
parameters = (n_epochs, lr, weight_decay, lh_var)

# inv_noprior = Inverse_model(noprior, excluded, saveframes=saveframes, plot=plot)
# inv_normal = Inverse_model(normprior, excluded, saveframes=saveframes, plot=plot)
# inv_gm = Inverse_model(gmprior, indices, saveframes=saveframes, plot=plot)
inv_hmmGauss = Inverse_model(hmmGaussprior, indices, saveframes=saveframes, plot=plot)

gen_timeseries(inv_hmmGauss, sequences, parameters, samples, view=True, init_pose='first pose', opt_pose='self',
               interpolate=True, sample_rate=sample_rate)

