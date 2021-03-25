import numpy as np
import torch
import timeit
from torch.distributions import MultivariateNormal


def log_likelihood(yhat, y, variance = 1):
    """Goal coordinates is 3xN tensor (in the same order as in the joint class)
    Takes joint coordinates as a 3xN tensor. where N is the number of goal joints
    Variance is chosen somewhat arbitrarily"""
    # #Method 1
    # mean = coordinates.flatten()
    # cov = torch.eye(mean.size(0)) * variance
    # goal = goal.flatten()
    # likelihood_model = MultivariateNormal(mean, covariance_matrix=cov)
    # return likelihood_model.log_prob(goal)

    #Method 2
    N = y.size(1)
    likelihoods = []
    cov = torch.eye(3) * variance
    for i in range(N):
        mean = yhat[:,i]
        goal_point = y[:,i]
        likelihood_model = MultivariateNormal(mean, covariance_matrix=cov)
        likelihoods.append(likelihood_model.log_prob(goal_point))
    return sum(likelihoods)

def normal_prior(mean, cov):
    """Has method prior_model.log_prob(*joint_angles as tensor*)
    Get mean and cov from normal_params.npz"""
    mean = torch.tensor(mean)
    cov = torch.tensor(cov)

    #Experimental
    cov = torch.diagonal(cov)
    cov = torch.diag(cov)

    prior_model = MultivariateNormal(mean, covariance_matrix = cov)
    return prior_model

def get_prior_model():
    """Returns means, covs and queries for different datasets of joint angles"""
    fname = r'./models/normal_params.npz'
    npzfile = np.load(fname, allow_pickle=True)
    means = npzfile['arr_0']
    covs = npzfile['arr_1']
    queries = npzfile['arr_2']
    return means, covs, queries


def Loss_Normal(pose, yhat, y, prior, variance):
    # print(joint_angles, coordinates, goal)
    likelihood = 0
    for i in range(len(yhat)):
        likelihood += log_likelihood(yhat[i], y[i], variance)
    prior = prior.log_prob(pose)
    LogLoss = -(likelihood + prior)
    #LogLoss = -prior
    return LogLoss

def Loss_Euclidian(yhat, y):
    loss = 0
    for i in range(len(yhat)):
        loss += torch.cdist(yhat[i].transpose(0, 1), y[i].transpose(0, 1), p=2.0)
    return loss

if __name__ == "__main__":
    #make a test
    variance = 1
    goal = torch.arange(3 * 20, dtype=torch.float32).reshape((3,-1))
    coordinates = torch.arange(3 * 20,dtype=torch.float32).reshape((3,-1))
    start1 = timeit.default_timer()
    Loss1 = log_likelihood(coordinates, goal, variance, 1)
    stop1 = timeit.default_timer()
    print('Time: ', stop1 - start1)
    start2 = timeit.default_timer()
    Loss2 = log_likelihood(coordinates, goal, variance, 0)
    stop2 = timeit.default_timer()
    print('Time: ', stop2 - start2)
    #Method 2 seems faster!
