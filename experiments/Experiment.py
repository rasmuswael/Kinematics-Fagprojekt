from inverse_kinematics.InverseKinematics import *
torch.manual_seed(1510)

sample_rate = 12

selected = get_fnames(["walk"])
data = parse_selected(selected, sample_rate=sample_rate, limit=1000)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1] - 3)]

dummy_joints, dummy_pose = dummy()

# excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes']
excluded = ['root', 'lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'rfoot', 'lfoot', 'head', 'rwrist', 'lwrist', 'rclavicle', 'lclavicle']

included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
mean, cov = compute_parameters_normal(truncate(X))
mean, cov = mean[indices], cov[indices, :][:, indices]

means, covs, weights = compute_gm_params(X, n_components=3, indices=indices)

noprior = ('noprior', None)
normprior = ('normal', normal_prior(mean, cov))
gmprior = ('gaussianmixture', gmm_prior(means, covs, weights))
# nfprior = ('normalizingflows', nf_prior(compute_NF(X, steps=100, indices=indices)))

# goal_joints = ['rfoot']
# pose = {'rfemur': [40, 0, 0]}

goal_joints = ['rfoot', 'lfoot']
pose = {'rfemur': [25, 0, 0], 'lfemur': [-25, 0, 0]}

goal = set_goal(goal_joints, pose)

saveframes, plot = True, True

n_epochs, lr, weight_decay, lh_var = 100, 6, 0, 1

inv_noprior = Inverse_model(noprior, indices, saveframes=saveframes, plot=plot)
inv_normal = Inverse_model(normprior, indices, saveframes=saveframes, plot=plot)
inv_gm = Inverse_model(gmprior, indices, saveframes=saveframes, plot=plot)
# inv_nf = Inverse_model(nfprior, indices, saveframes=saveframes, plot=plot)

inv_noprior.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)
inv_normal.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)
inv_gm.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)
# inv_nf.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)

# Frames = [inv_noprior.frames, inv_normal.frames, inv_gm.frames, inv_nf.frames]
Frames = [inv_noprior.frames, inv_normal.frames, inv_gm.frames]


for frames in Frames:
    v = Viewer(dummy_joints_np(), frames)
    v.run()
