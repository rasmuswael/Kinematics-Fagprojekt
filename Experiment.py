from InverseKinematics import *
torch.manual_seed(1510)

selected = get_fnames(["walk"])
data = parse_selected(selected, relative_sample_rate=8, limit=10000)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1] - 3)]

dummy_joints, dummy_pose = dummy()

excluded = ['root','lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes']
included, indices = exclude(excluded, return_indices=True)
mean, cov = compute_parameters_normal(truncate(X))

mean, cov = mean[indices], cov[indices, :][:, indices]

noprior = ('noprior', None)
normprior = ('normal', normal_prior(mean, cov))
n_components, n_features = 16, len(indices)
X = remove_excluded(truncate(X), indices, type='numpy')
mu_init = X[np.random.choice(X.shape[0], size = n_components), :]
gm = GaussianMixture(n_components=n_components, n_features=n_features, mu_init=torch.tensor([mu_init]))
gm.fit(torch.tensor(X), n_iter=250)
gmprior = ('gaussian', gm)

# goal_joints = ['rfoot']
# pose = {'rfemur': [40, 0, 0]}

goal_joints = ['rfoot', 'rhand']
pose = {'rfemur': [40, 0, 0], 'rhumerus': [90,0,0]}


goal = set_goal(goal_joints, pose)

inv_normal = Inverse_model(gmprior, excluded, saveframes=True, plot=False)
inv_normal.inverse_kinematics(goal)
v = Viewer(dummy_joints, inv_normal.frames)
v.run()
# inverse_kinematics(goal)