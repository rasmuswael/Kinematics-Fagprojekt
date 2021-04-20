from get_subjects import *
from dummy import dummy
from scipy.spatial.transform import Rotation
import torch
from torch.distributions import MultivariateNormal
from scipy.stats import circmean, circvar
#from gmm_torch.gmm import GaussianMixture
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm


def truncate(X, root_limit=[]):
    '''Remove translation from X coloumns first'''
    X = (X + 180) % 360 - 180
    joints, pose, dof = dummy(return_dof=True)

    if not len(root_limit):
        root_limit = np.array([[-30., 30.],
                               [0., 0.],
                               [-30., 30.]])

    i = 0
    for joint in list(pose.keys()):
        if joint == 'root':
            limits = root_limit
        else:
            limits = joints[joint].limits
        for axis in dof[joint]:
            low, high = limits[axis, 0], limits[axis, 1]
            X[:,i] = np.clip(X[:,i], low, high)
            i +=1
    return(X)


### Helper functions
def exclude(excluded=None, return_indices=True, root_exclude=[1]):
    '''Orientation of the skeleton is excluded by default. The index is included if return_indices is called'''
    _, dummy_pose, dof = dummy(return_dof=True)

    if excluded == None:
        excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes']

    included = list(set(dummy_pose.keys()) - set(excluded))

    if return_indices:
        exc_indices = root_exclude
        count = 0
        for j, d in dof.items():
            if j in excluded:
                for i in range(count, count + len(d)):
                    exc_indices.append(i)
            count += len(d)
        indices = np.delete(np.arange(count), exc_indices)
        return included, indices
    return included


def array2pose(A, indices=[], type='torch'):
    _, dummy_pose, dof = dummy(return_dof=True)

    if len(indices):
        if type == 'torch':
            B = torch.zeros(59).float()
            B[indices] = A.float()
            A = B
        elif type == 'numpy':
            B = np.zeros(59)
            B[indices] = A
            A = B

    idx = 0
    for j, d in dof.items():
        for i in range(len(d)):
            if j == 'root':
                dummy_pose[j][i+3] = A[idx + i]
            else:
                dummy_pose[j][i] = A[idx + i]
        idx += len(d)
    return dummy_pose


def pose2array(pose, type='torch'):
    A = pose['root'][3:]
    angle_list = list(pose.values())

    for angles in angle_list[1:]:
        if type == 'torch':
            A = torch.cat([A,angles])
        elif type == 'numpy':
            A = np.concatenate([A,angles])
    return A


def array2motions(X, indices=np.arange(59)):
    motions = []
    for x in X:
        motions.append((array2pose(torch.tensor(x), indices)))
    return motions


def motions2array(motions):
    X = np.array(pose2array(motions[0]))
    for pose in motions[1:]:
        X = np.vstack((X,np.array(pose2array(pose))))
    return X



def remove_excluded(A, indices, datatype='array', type='torch'):
    '''Assumes that nothing has been been removed'''
    if datatype == 'dict':
        A = pose2array(A, type=type)

    if type == 'torch':
        shape = A.size()
        A = A.flatten()
        numreps = A.size(0) // 59
        B = torch.empty(0)
        for rep in range(numreps):
            B = torch.cat((B, A[indices + rep * 59]))
    if type == 'numpy':
        shape = A.shape
        A = A.flatten()
        numreps = A.shape[0] // 59
        B = np.empty(0)
        for rep in range(numreps):
            B = np.concatenate((B, A[indices + rep * 59]))
    if len(shape) - 1:
        B = B.reshape((numreps, len(indices)))

    if datatype == 'dict':
        return array2pose(B, indices, type=type)
    return B


def draw_cluster(mean, indices=[], type='torch'):
    joints, pose, dof = dummy(return_dof=True)
    mean_pose = array2pose(mean, indices, type)
    joints['root'].set_motion(mean_pose)
    joints['root'].draw()
###

def compute_circ_params(X):
    '''Remove translation from X coloumns first'''
    X = np.deg2rad(X)
    cmean = circmean(X, axis=0)
    cvar = circvar(X, axis=0)
    return np.rad2deg(cmean), np.rad2deg(cvar)


def compute_parameters_normal(X):
    '''Remove translations X coloumns first'''
    mean = X.mean(0)
    cov = np.cov(X, rowvar=False)
    return mean, cov


# def compute_gm(X, n_components, indices=np.arange(59), initiliaze_mu=True):
#     n_features = len(indices)
#     X = remove_excluded(truncate(X), indices, type='numpy')
#     if initiliaze_mu:
#         mu_init = X[np.random.choice(X.shape[0], size=n_components), :]
#         gm = GaussianMixture(n_components=n_components, n_features=n_features, mu_init=torch.tensor([mu_init]))
#     else:
#         gm = GaussianMixture(n_components=n_components, n_features=n_features)
#     gm.fit(torch.tensor(X), n_iter=250)
#     return gm


def compute_gm_params(X, n_components, indices=np.arange(59), covariance_type='full'):
    X = remove_excluded(truncate(X), indices, type='numpy')
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type).fit(X)
    return gm.means_, gm.covariances_, gm.weights_


def compute_hmmGauss(X, lengths, n_components, indices=np.arange(59), covariance_type='full'):
    X = remove_excluded(truncate(X), indices, type='numpy')
    model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type).fit(X, lengths=lengths)
    return model


### EXPERIMENTAL ###
def angle2cartesian(X):
    X = np.deg2rad(X)
    Xcos, Xsin = np.cos(X), np.sin(X)
    return np.hstack((Xcos, Xsin))


def cartesian2angle(X):
    shape = X.shape
    onedim = not len(shape) - 1
    if onedim:
        ncols = shape[0]
        Xcos, Xsin = X[:int(ncols / 2)], X[int(ncols / 2):]
    else:
        ncols = shape[1]
        Xcos, Xsin = X[:, :int(ncols / 2)], X[:, int(ncols / 2):]
    X = np.zeros(Xcos.shape)
    for i in range(int(ncols / 2)):
        if onedim:
            X[i] = np.rad2deg(np.arctan2(Xsin[i], Xcos[i]))
        else:
            X[:, i] = np.rad2deg(np.arctan2(Xsin[:, i], Xcos[:, i]))
    return X


###
if __name__ == "__main__":
    # output_file = "./models/normal_params"
    # means = []
    # covs = []
    # query_list = [["walk"], ["dance"], ["walk", "run"]]
    # for queries in query_list:
    #     mean, cov = compute_parameters_normal(queries)
    #     means.append(mean)
    #     covs.append(cov)
    # np.savez(output_file, np.array(means), np.array(covs), np.array(query_list))
    # means, covs, queries = get_prior_model()
    # mean, cov = means[0], covs[0]
    # Lets try visualizing the circular mean
    selected = get_fnames(["walk"])
    data = parse_selected(selected, sample_rate=8, limit=2000)
    X, y = gather_all_np(data)
    X = X[:, :(X.shape[1] - 3)]

    dummy_joints, dummy_pose = dummy()

    included, indices = exclude(return_indices=True)
    mean, cov = compute_parameters_normal(truncate(X))

    truncmean, trunccov = np.zeros(mean.shape), np.zeros(cov.shape)
    truncmean[indices], trunccov[indices, indices] = mean[indices], cov[indices, indices]
    mean, cov = torch.tensor(truncmean), torch.tensor(trunccov)
    # Draw prior
    mean_pose = array2pose(mean)

    dummy_joints['root'].set_motion(mean_pose)
    dummy_joints['root'].draw()

    # Ignore orientation
    # X[:, 1] = np.zeros(X.shape[0])

    cmean, _ = compute_circ_params(X)
    trunccmean, _ = compute_circ_params(truncate(X))

    # Xcart = angle2cartesian(X)
    # Xangle = cartesian2angle(Xcart)
    # cart_mean = np.mean(Xcart, axis=0)
    # angle_mean = cartesian2angle(cart_mean)
    # cart_cov = np.var(Xcart, axis=0)
    # angle_cov = cartesian2angle(cart_cov)
    mean, cov = compute_parameters_normal(X)
    truncmean, trunccov = compute_parameters_normal(truncate(X))

    draw_cluster(cmean)
    draw_cluster(trunccmean)
    draw_cluster(truncmean)
    #draw_cluster(angle_mean)
    #draw_cluster(mean)