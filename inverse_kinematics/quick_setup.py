from inverse_kinematics.TimeSeries import *
import pickle

def quick_setup(queries, data_pool="manual", sample_rate=1, limit=None, excluded=[], returny=False):
    """Write an elaborate docstring here"""
    sample_rate = sample_rate

    if data_pool == 'manual':
        selected = get_manual_names(queries)
    elif data_pool == 'all':
        selected = get_fnames(queries)
    else:
        selected = data_pool

    data = parse_selected(selected, sample_rate=sample_rate, limit=limit)
    len_array = get_lengths_np(data)
    X, y = gather_all_np(data)
    X = X[:, :(X.shape[1] - 3)]

    if not len(excluded):
        excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist',
                    'lwrist', 'rclavicle', 'lclavicle']

    included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
    print(f"The following joints are included in the model:\n {included}")
    if returny:
        return (X, y), len_array, indices
    return X, len_array, indices


def get_prior(type, parameters=None):
    """Pass the type and the appropriate parameters to get the prior tuple. See train_prior for more information"""
    if type == 'noprior':
        return ('noprior', None)
    elif type == 'normal':
        return ('normal', normal_prior(*parameters))
    elif type == 'gaussianmixture':
        return ('gaussianmixture', gmm_prior(*parameters))
    elif type == 'hmmGauss':
        model = parameters
        return ('hmmGauss', (gmm_prior(model.means_, model.covars_, torch.tensor([1/len(model.means_)]*len(model.means_))), model))
        # return ('hmmGauss', (gmm_prior(model.means_, model.covars_, torch.tensor(model.startprob_)), model))


def train_prior(X, indices, type='noprior',hyperparameters=[], savepath=None):
    """Pass array X, indices, type and hyperparameters to obtain parameters for the prior model
    :arg X, array of training data
    :arg indices, indices of the included joint angles
    :arg hyperparameters, hyperparameters for the prior distribution
    :arg savepath, savepath for the model. If None, model is not saved.
    :return parameters, parameters for the prior model"""
    if type == 'normal':
        parameters = compute_parameters_normal(remove_excluded(truncate(X), indices, type='numpy'))
        if savepath is not None:
            with open(f"./models/{savepath}.pkl", "wb") as file:
                pickle.dump(parameters, file)
        return parameters
    elif type == 'gaussianmixture':
        n_components, covariance_type = hyperparameters
        parameters = compute_gm_params(X, n_components=n_components, indices=indices, covariance_type=covariance_type)
        if savepath is not None:
            with open(f"./models/{savepath}.pkl", "wb") as file:
                pickle.dump(parameters, file)
        return parameters
    elif type == 'hmmGauss':
        n_states, len_array, covariance_type = hyperparameters
        model = compute_hmmGauss(X, len_array, n_components=n_states, indices=indices, covariance_type='full')
        if savepath is not None:
            with open(f"./models/{savepath}.pkl", "wb") as file:
                pickle.dump(model, file)
        return model

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

if __name__ == '__main__':
    X, len_array, indices = quick_setup(['run'], data_pool="manual", sample_rate=6, limit=None)
    print(X.shape)