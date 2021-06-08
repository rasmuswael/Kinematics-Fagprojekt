

def fit_NF(X):
    X = remove_excluded(truncate(X), indices, type='numpy')

    base_dist = dist.Normal(torch.zeros(2), torch.ones(2))
    spline_transform = T.spline_coupling(2, count_bins=16)
    flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])