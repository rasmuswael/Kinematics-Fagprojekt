import torch
import timeit
from torch.distributions import MultivariateNormal
import numpy as np

def log_likelihood(yhat, y, variance = 1):
    """Goal coordinates is 3xN tensor (in the same order as in the joint class)
    Takes joint coordinates as a 3xN tensor, where N is the number of goal joints
    Variance is chosen somewhat arbitrarily
    :arg yhat, 3d coordinate tensor for the goal joint
    :arg y, 3d coordinate tensor for the goal coordinate corresponding the goal joint
    :arg variance, likelihood variance. Defaults to 1"""

    #Method 2
    N = y.size(1)
    return -sum([(((y[:,i]-yhat[:,i])**2).sum()/variance) for i in range(N)])
    #cov = torch.eye(3) * variance
    #return sum([MultivariateNormal(yhat[:,i], covariance_matrix=cov).log_prob(y[:,i]) for i in range(N)])


def normal_prior(mean, cov):
    """Has method prior_model.log_prob(*joint_angles as tensor*)
    Get mean and cov from normal_params.npz"""
    mean = torch.tensor(mean)
    cov = torch.tensor(cov)

    prior_model = MultivariateNormal(mean, covariance_matrix = cov)
    return prior_model


class gmm_prior:
    """gmm_prior class"""
    def __init__(self, means, covs, weights):
        self.weights = torch.tensor(weights)
        self.means = torch.tensor(means)
        self.covs = torch.tensor(covs)
        self.n_components = len(weights)
        self.gaussians = [MultivariateNormal(self.means[i], covariance_matrix=self.covs[i]) for i in range(self.n_components)]

    def log_prob(self, pose):
        # return torch.logsumexp(torch.tensor([self.gaussians[i].log_prob(pose) + torch.log(self.weights[i]+1e-7)for i in range(self.n_components)]), 0)
        return sum([self.gaussians[i].log_prob(pose) + torch.log(self.weights[i]+1e-7) for i in range(self.n_components)])

    def sample(self, N):
        samples = torch.tensor([])
        n_features = self.means.shape[1]
        for j in range(N):
            i = np.random.choice(np.arange(self.n_components), p=self.weights.tolist())
            samples = torch.cat((samples, self.gaussians[i].rsample()))
        return samples.reshape((N,n_features))


class nf_prior:
    """Normalizing flows prior (unused)"""
    def __init__(self, nf_model):
        self.nf_dist, self.mean, self.std = nf_model

    def log_prob(self, pose):
        return self.nf_dist.log_prob((pose - self.mean) / self.std)


class euclidian_Loss:
    def Loss(self, yhat, y, pose):
        loss = 0
        for i in range(yhat.size(1)):
            loss += torch.cdist(yhat[:,i].transpose(0, 1), y[:,i].transpose(0, 1), p=2.0)
        return loss


class posterior_Loss:
    """The main posterior_Loss class. Every loss function class inherits from this class."""
    def __init__(self, prior, likelihood_variance=1e-2):
        self.prior_type, self.prior_model = prior
        self.lh_var = likelihood_variance


    def Loss_likelihood(self, yhat, y):
        return log_likelihood(yhat, y, self.lh_var)


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


class nfprior_Loss(posterior_Loss):
    def __init__(self, nf_model, likelihood_variance):
        super().__init__(('Normalizing Flows model prior', nf_model), likelihood_variance)


    def Loss(self, yhat, y, pose):
        return -(super().Loss_likelihood(yhat, y) + self.prior_model.log_prob(pose))


def get_loss_func(prior, likelihood_variance):
    prior_type, prior_model = prior
    if prior_type == None:
        return euclidian_Loss()
    elif prior_type == 'noprior':
        return noprior_Loss(likelihood_variance)
    elif prior_type == 'normal':
        return normalprior_Loss(prior_model, likelihood_variance)
    elif prior_type == 'gaussianmixture':
        return gmprior_Loss(prior_model, likelihood_variance)
    elif prior_type == 'hmmGauss':
        gm_prior, hmm_model = prior_model
        return hmmGaussprior_Loss(hmm_model, gm_prior, likelihood_variance)
    elif prior_type == 'normalizingflows':
        return nfprior_Loss(prior_model, likelihood_variance)
