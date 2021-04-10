from InverseKinematics import *
torch.manual_seed(1510)

selected = get_fnames(["walk"])
data = parse_selected(selected, sample_rate=8, limit=10000)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1] - 3)]

dummy_joints, dummy_pose = dummy()

# excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes']
excluded = ['root', 'lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'rfoot', 'lfoot', 'head', 'rwrist', 'lwrist', 'rclavicle', 'lclavicle']

included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
mean, cov = compute_parameters_normal(truncate(X))
mean, cov = mean[indices], cov[indices, :][:, indices]

gm = compute_gm(X, n_components=8, indices=indices, initiliaze_mu=True)

noprior = ('noprior', None)
normprior = ('normal', normal_prior(mean, cov))
gmprior = ('gaussian', gm)

# goal_joints = ['rfoot']
# pose = {'rfemur': [40, 0, 0]}

goal_joints = ['rfoot', 'lfoot']
pose = {'rfemur': [25, 0, 0], 'lfemur': [-25, 0, 0]}

goal = set_goal(goal_joints, pose)

saveframes, plot = True, True

n_epochs, lr, weight_decay, lh_var = 75, 6, 0, 1

inv_noprior = Inverse_model(noprior, excluded, saveframes=saveframes, plot=plot)
inv_normal = Inverse_model(normprior, excluded, saveframes=saveframes, plot=plot)
inv_gm = Inverse_model(gmprior, excluded, saveframes=saveframes, plot=plot)

inv_noprior.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)
inv_normal.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)
inv_gm.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)

Frames = [inv_noprior.frames, inv_normal.frames, inv_gm.frames]

for frames in Frames:
    v = Viewer(dummy_joints, frames)
    v.run()
