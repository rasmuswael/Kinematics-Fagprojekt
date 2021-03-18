import numpy as np
import torch
import timeit
from torch.distributions import MultivariateNormal


def log_likelihood(coordinates, goal, variance = 1):
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
    N = goal.size(1)
    likelihoods = []
    cov = torch.eye(3) * variance
    for i in range(N):
        mean = coordinates[:,i]
        goal_point = goal[:,i]
        likelihood_model = MultivariateNormal(mean, covariance_matrix=cov)
        likelihoods.append(likelihood_model.log_prob(goal_point))
    return sum(likelihoods)

def normal_prior(mean, cov):
    """Has method prior_model.log_prob(*joint_angles as tensor*)
    Get mean and cov from normal_params.npz"""
    mean = torch.tensor(mean)
    cov = torch.tensor(cov)

    # #Experimental
    # cov = torch.diagonal(cov)
    # cov = torch.diag(cov) * 100

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


def Loss_V1(joint_angles, coordinates, goal, prior_model, variance):
    # print(joint_angles, coordinates, goal)
    print(coordinates, goal)
    likelihood = log_likelihood(coordinates, goal, variance)
    prior = prior_model.log_prob(joint_angles)
    print(likelihood, prior)
    LogLoss = -(likelihood + prior)
    # LogLoss = -log_likelihood(coordinates, goal, variance)
    # LogLoss = -prior
    return LogLoss

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
