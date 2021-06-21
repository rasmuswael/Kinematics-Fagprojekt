from inverse_kinematics.get_subjects import *
from inverse_kinematics.dummy import dummy, dummy_joints_np
import torch
from scipy.stats import circmean, circvar
from sklearn.mixture import GaussianMixture
from transforms3d.euler import euler2mat, mat2euler
from scipy.spatial.transform import Rotation as R

import pyro
import pyro.distributions as dist
import pyro.distributions.transforms as T

from hmmlearn import hmm


def truncate(X, root_limit=[]):
    """Truncates the numpy matrix of angles to prepare for statistics, and removes orientation.
    Remove translation from X coloumns first"""
    joints, pose, dof = dummy(return_dof=True)

    root = R.from_euler('xyz', X[:,:3],degrees=True)
    c = root.as_euler('zxy', degrees=True)
    X[:,0] = c[:,1]
    X[:,1] = 0
    X[:,2] = c[:,0]
    X = (X + 180) % 360 - 180

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
    return X

def standardize(X):
    mean, std = X.mean(0), X.std(0)
    return (X-mean)/std, mean, std

def compute_parameters_normal(X):
    '''Remove translations X coloumns first'''
    mean = X.mean(0)
    cov = np.cov(X, rowvar=False)
    return mean, cov


def compute_gm_params(X, n_components, indices=np.arange(59), covariance_type='full'):
    X = remove_excluded(truncate(X), indices, type='numpy')
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type).fit(X)
    return gm.means_, gm.covariances_, gm.weights_


def compute_hmmGauss(X, lengths, n_components, indices=np.arange(59), covariance_type='full'):
    X = remove_excluded(truncate(X), indices, type='numpy')
    model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type).fit(X, lengths=lengths)
    return model


def compute_NF(X, steps, indices=np.arange(59), lr=1e-2):
    X = remove_excluded(truncate(X), indices, type='numpy')
    X, mean, std = standardize(X)
    n_features = X.shape[1]

    base_dist = dist.Normal(torch.zeros(n_features), torch.ones(n_features))
    spline_transform = T.spline_coupling(n_features, hidden_dims=[n_features * 2] * 128, split_dim=None, count_bins=32)
    flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])

    dataset = torch.from_numpy(X).float()
    optimizer = torch.optim.Adam(spline_transform.parameters(), lr=lr)
    for step in range(steps + 1):
        optimizer.zero_grad()
        loss = -flow_dist.log_prob(dataset).mean()
        loss.backward()
        optimizer.step()
        flow_dist.clear_cache()

        if step % 10 == 0:
            print('step: {}, loss: {}'.format(step, loss.item()))
    return (flow_dist, torch.tensor(mean, dtype=torch.float), torch.tensor(std, dtype=torch.float))


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


def convertpose(pose, type):
    """Will convert the pose from torch to numpy or numpy to torch. The type argument is the current type"""
    if type == 'torch':
        return {k: v.detach().numpy() for k,v in pose.items()}
    if type == 'numpy':
        return {{k: torch.tensor(v) for k,v in pose.items()}}


def array2pose(A, indices=[], type='torch'):
    _, dummy_pose, dof = dummy(return_dof=True)

    if type == 'numpy':
        dummy_pose = convertpose(dummy_pose, 'torch')

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


def array2motions(X, indices=np.arange(59), type='torch'):
    """Assumes that X is a numpy array"""
    motions = []
    if type=='torch':
        X = torch.tensor(X)
    for x in tqdm(X):
        motions.append((array2pose(x, indices, type=type)))
    return motions


def motions2array(motions, type='torch'):
    X = np.array(pose2array(motions[0], type=type))
    for pose in motions[1:]:
        X = np.vstack((X,np.array(pose2array(pose, type=type))))
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


def draw_cluster(mean, indices=[], type='torch', fig=None):
    joints, pose, dof = dummy(return_dof=True)
    mean_pose = array2pose(mean, indices, type)
    if type=='numpy':
        mean_pose = convertpose(mean_pose, 'numpy')
    joints['root'].set_motion(mean_pose)
    return joints['root'].draw(fig=fig)
### END OF HELPER FUNCTIONS