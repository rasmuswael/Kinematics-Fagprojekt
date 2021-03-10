import torch
from torch.distributions import MultivariateNormal


def likelihood(coordinates, goal, variance):
    """Goal coordinates is 3xN tensor (in the same order as in the joint class)
    Takes joint coordinates as a 3xN tensor. where N is the number of goal joints"""
    # #Method 1
    # mean = coordinates.flatten()
    # cov = torch.eye(mean.size(0)) * variance
    # goal = goal.flatten()
    # likelihood_model = MultivariateNormal(mean, covariance_matrix=cov)
    # return likelihood_model.log_prob(goal)

    #Method 2
    N = goal.size(1)
    likelihoods = []
    for i in range(N):
        mean = coordinates[:,i]
        cov = torch.eye(3) * variance
        goal_point = goal[:,i]
        likelihood_model = MultivariateNormal(mean, covariance_matrix=cov)
        likelihoods.append(likelihood_model.log_prob(goal_point))
    return sum(likelihoods) / N

if __name__ == "__main__":
    #make a test
    variance = 1
    goal = torch.arange(6, dtype=torch.float32).reshape((3,-1))
    coordinates = torch.arange(6,dtype=torch.float32).reshape((3,-1))
    Loss = likelihood(coordinates, goal,variance)
