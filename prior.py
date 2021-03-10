from get_subjects import *
import torch
from torch.distributions import MultivariateNormal

# selected = get_fnames(["walk"])
# data = parse_selected(selected)
# X, y = gather_all_np(data)
#
# # Remove translations
# X = X[:, :(X.shape[1]-3)]

def compute_parameters_normal(queries):
    selected = get_fnames(queries)
    data = parse_selected(selected)
    X, y = gather_all_np(data)
    # Remove translations
    X = X[:, :(X.shape[1] - 3)]
    mean = X.mean(0)
    cov = np.cov(X, rowvar=False)
    return mean, cov

def normal_prior(mean, cov):
    # Has method prior_model.log_prob(*joint_angles as tensor*)
    mean = torch.tensor(mean)
    cov = torch.tensor(cov)

    prior_model = MultivariateNormal(mean, covariance_matrix = cov)
    return prior_model

if __name__ == "__main__":
    output_file = "./models/normal_params"
    means = []
    covs = []
    query_list = [["walk"], ["dance"], ["walk", "run"]]
    for queries in query_list:
        mean, cov = compute_parameters_normal(queries)
        means.append(mean)
        covs.append(cov)
    np.savez(output_file, np.array(means), np.array(covs), np.array(query_list))
