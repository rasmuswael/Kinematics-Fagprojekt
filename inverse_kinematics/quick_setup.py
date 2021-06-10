from inverse_kinematics.TimeSeries import *
import pickle

def quick_setup(queries, data_pool="manual", sample_rate=1, limit=None, excluded=[]):
    """Write an elaborate docstring here"""
    sample_rate = sample_rate

    if data_pool == 'manual':
        selected = get_manual_names(queries)
    else:
        selected = get_fnames(queries)

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
        model = parameters
        return ('hmmGauss', (gmm_prior(model.means_, model.covars_, torch.tensor([1/len(model.means_)]*len(model.means_))), model))


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
        n_states, len_array, covariance_type = hyperparameters
        model = compute_hmmGauss(X, len_array, n_components=n_states, indices=indices, covariance_type='full')
        return get_prior(type, model)


#new version - returns parameters
def train_prior(X, indices, type='noprior',hyperparameters=[], path, save=True):
    """Write an elaborate docstring here"""
    if type == 'noprior':
        return
    elif type == 'normal':
        parameters = compute_parameters_normal(remove_excluded(truncate(X), indices, type='numpy'))
        if save==True:
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(parameters, file)
        return get_prior(type, parameters)

    elif type == 'gaussianmixture':
        n_components, covariance_type = hyperparameters
        parameters = compute_gm_params(X, n_components=n_components, indices=indices, covariance_type=covariance_type)
        if save==True:
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(parameters, file)
        return get_prior(type, parameters)

    elif type == 'hmmGauss':
        n_states, len_array, covariance_type = hyperparameters
        model = compute_hmmGauss(X, len_array, n_components=n_states, indices=indices, covariance_type='full')
        if save == True:
            with open(f"{path}.pkl", "wb") as file:
                pickle.dump(model, file)
        return get_prior(type, model)

def load_param(path, type):
    """
    Decodes model pickle-file
    :param path: str, path to model in ./models
    :param type: str, type of model
    :return: parameters
    """

    if type == 'noprior':
        return

    elif type == 'normal':
        with open(f"{path}.pkl", "rb") as file:
            parameters = pickle.load(file)
        return parameters

    elif type == 'gaussianmixture':
        with open(f"{path}.pkl", "rb") as file:
            parameters = pickle.load(file)
        return parameters

    elif type == 'hmmGauss':
        with open(f"{path}.pkl", "rb") as file:
            parameters = pickle.load(file)
        return parameters