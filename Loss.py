import torch
import timeit
from torch.distributions import MultivariateNormal


def log_likelihood(yhat, y, variance = 1):
    """Goal coordinates is 3xN tensor (in the same order as in the joint class)
    Takes joint coordinates as a 3xN tensor. where N is the number of goal joints
    Variance is chosen somewhat arbitrarily"""

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
    # cov = torch.diagonal(cov)
    # cov = torch.diag(cov)

    prior_model = MultivariateNormal(mean, covariance_matrix = cov)
    return prior_model


class gmm_prior:
    def __init__(self, means, covs, weights):
        self.weights = torch.tensor(weights)
        self.means = torch.tensor(means)
        self.covs = torch.tensor(covs)
        self.n_components = len(weights)
        self.gaussians = [MultivariateNormal(self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_components)]

    def log_prob(self, pose):
        prior_prob = sum([self.gaussians[i].log_prob(pose) * self.weights[i] for i in range(self.n_components)])
        return prior_prob



class euclidian_Loss:
    def Loss(self, yhat, y, pose):
        loss = 0
        for i in range(len(yhat)):
            loss += torch.cdist(yhat[i].transpose(0, 1), y[i].transpose(0, 1), p=2.0)
        return loss


class posterior_Loss:
    def __init__(self, prior, likelihood_variance=1e-2):
        self.prior_type, self.prior_model = prior
        self.lh_var = likelihood_variance


    def Loss_likelihood(self, yhat, y):
        likelihood = 0
        for i in range(len(yhat)):
            likelihood += log_likelihood(yhat[i], y[i], self.lh_var)
        return likelihood / len(yhat)


class noprior_Loss(posterior_Loss):
    def __init__(self, likelihood_variance):
        super().__init__(('No prior', None), likelihood_variance)


    def Loss(self, yhat, y, pose):
        return -super().Loss_likelihood(yhat, y)


class normalprior_Loss(posterior_Loss):
    def __init__(self, normal_model, likelihood_variance):
        super().__init__(('Normal prior', normal_model), likelihood_variance)


    def Loss(self, yhat, y, pose):
        return -(super().Loss_likelihood(yhat, y) + self.prior_model.log_prob(pose))


class gmprior_Loss(posterior_Loss):
    def __init__(self, gm_model, likelihood_variance):
        super().__init__(('Gaussian Mixture Model prior', gm_model), likelihood_variance)


    def Loss(self, yhat, y, pose):
        return -(super().Loss_likelihood(yhat, y) + self.prior_model.log_prob(pose))


class hmmGaussprior_Loss(posterior_Loss):
    def __init__(self, hmm_model, gm_model, likelihood_variance):
        self.hmm_model = hmm_model
        super().__init__(('Hidden Markov Model, Gaussian prior', gm_model), likelihood_variance)


    def update_weights(self, prev_result):
        self.prior_model.weights = torch.tensor(self.hmm_model.predict_proba(prev_result)[0])


    def Loss(self, yhat, y, pose):
        return -(super().Loss_likelihood(yhat, y) + self.prior_model.log_prob(pose))


def get_loss_func(prior, likelihood_variance):
    prior_type, prior_model = prior
    if prior_type == None:
        func = euclidian_Loss()
    elif prior_type == 'noprior':
        func = noprior_Loss(likelihood_variance)
    elif prior_type == 'normal':
        func = normalprior_Loss(prior_model, likelihood_variance)
    elif prior_type == 'gaussianmixture':
        func = gmprior_Loss(prior_model, likelihood_variance)
    elif prior_type == 'hmmGauss':
        gm_prior, hmm_model = prior_model
        func = hmmGaussprior_Loss(hmm_model, gm_prior, likelihood_variance)
    return func


# def Loss_Likelihood(yhat, y, variance):
#     likelihood = 0
#     for i in range(len(yhat)):
#         likelihood += log_likelihood(yhat[i], y[i], variance)
#     return likelihood / len(yhat)
# def Loss_Normal(pose, yhat, y, normal_prior, variance):
#     # print(joint_angles, coordinates, goal)
#     likelihood = Loss_Likelihood(yhat, y, variance)
#     prior = normal_prior.log_prob(pose)
#     LogLoss = -(likelihood + prior)
#     #LogLoss = -prior
#     return LogLoss
#
#
# def Loss_GM(pose, yhat, y, gmm_prior, variance):
#     likelihood = Loss_Likelihood(yhat, y, variance)
#     prior = gmm_prior.log_prob(pose)
#     LogLoss = -(likelihood + prior)
#     return LogLoss
#
#
# def Loss_Euclidian(yhat, y):
#     loss = 0
#     for i in range(len(yhat)):
#         loss += torch.cdist(yhat[i].transpose(0, 1), y[i].transpose(0, 1), p=2.0)
#     return loss
#
#
# def Loss_HMMGauss(pose, yhat, y, gmm_prior, model, variance):
#     gmm_prior.weights = torch.tensor(model.predict_proba(pose))
#     LogLoss = Loss_GM(pose, yhat, y, gmm_prior, variance)
#     return LogLoss
#
#
# def Loss(yhat, y, prior_model, prior_type, pose, lh_var=1e-2):
#     if prior_type == None:
#         loss = Loss_Euclidian(yhat, y)
#     elif prior_type == 'noprior':
#         loss = -Loss_Likelihood(yhat, y, lh_var)
#     elif prior_type == 'normal':
#         loss = Loss_Normal(pose, yhat, y, prior_model, lh_var)
#     elif prior_type == 'gaussian':
#         loss = Loss_GM(pose, yhat, y, prior_model, lh_var)
#     elif prior_type == 'hmmGauss':
#         gmm_prior, model = prior_model
#         loss = Loss_HMMGauss(pose, yhat, y, gmm_prior, model, variance)
#     return loss




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
