from inverse_kinematics.TimeSeries import *


def quick_setup(selected, data_pool="manual", sample_rate=1, limit=None, excluded=[]):
    """Write an elaborate docstring here"""
    sample_rate = sample_rate

    if data_pool == 'manual':
        selected = get_manual_names(selected)
    else:
        selected = get_fnames(selected)

    data = parse_selected(selected, sample_rate=sample_rate, limit=limit)
    len_array = get_lengths_np(data)
    X, y = gather_all_np(data)
    X = X[:, :(X.shape[1] - 3)]

    if not len(excluded):
        excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist',
                    'lwrist', 'rclavicle', 'lclavicle']

    included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
    print(f"The following joints are included in the model:\n {included}")
    return X, len_array, indices


def get_prior(type, parameters=None):
    """Write an elaborate docstring here"""
    if type == 'noprior':
        return ('noprior', None)
    elif type == 'normal':
        return ('normal', normal_prior(*parameters))
    elif type == 'gaussianmixture':
        return ('gaussianmixture', gmm_prior(*parameters))
    elif type == 'hmmGauss':
        model, n_states = parameters
        return ('hmmGauss', (gmm_prior(model.means_, model.covars_, torch.zeros(n_states)), model))

def train_prior(X, indices, type='noprior',hyperparameters=[]):
    """Write an elaborate docstring here"""
    if type == 'noprior':
        return get_prior(type)
    elif type == 'normal':
        parameters = compute_parameters_normal(remove_excluded(truncate(X), indices, type='numpy'))
        return get_prior(type, parameters)
    elif type == 'gaussianmixture':
        n_components, covariance_type = hyperparameters
        parameters = compute_gm_params(X, n_components=n_components, indices=indices, covariance_type=covariance_type)
        return get_prior(type, parameters)
    elif type == 'hmmGauss':
        n_states, covariance_type = hyperparameters
        model = compute_hmmGauss(X, len_array, n_components=n_states, indices=indices, covariance_type='full')
        return get_prior(type, (model, n_states))

goal_joints = ['rfoot', 'lfoot']

examples = selected
# examples = {'104': [('104_56','')]}

np.random.seed(seed)
samples = get_motion_samples(examples, 100, 2, sample_rate=sample_rate)
sequences, trunc_samples = get_goal_sequences(goal_joints, samples, indices, return_trunc_samples=True)

saveframes, plot = True, False

n_epochs, lr, weight_decay, lh_var = 100, 1e-1 * sample_rate, 0, 1e-2
parameters = (n_epochs, lr, weight_decay, lh_var)

# inv_noprior = Inverse_model(noprior, excluded, saveframes=saveframes, plot=plot)
# inv_normal = Inverse_model(normprior, excluded, saveframes=saveframes, plot=plot)
# inv_gm = Inverse_model(gmprior, indices, saveframes=saveframes, plot=plot)
inv_hmmGauss = Inverse_model(hmmGaussprior, indices, saveframes=saveframes, plot=plot)


gen_timeseries(inv_hmmGauss, sequences, parameters, trunc_samples, view=True, init_pose='first pose', opt_pose='self',
               interpolate=True, sample_rate=sample_rate)

