from InverseKinematics import *
torch.manual_seed(1510)

selected = get_fnames(["walk"])
data = parse_selected(selected, sample_rate=2, limit=15000)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1] - 3)]

dummy_joints, dummy_pose = dummy()

# excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes']
excluded = ['root', 'lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'rfoot', 'lfoot']

included, indices = exclude(excluded, return_indices=True)
# mean, cov = compute_parameters_normal(truncate(X))
# mean, cov = mean[indices], cov[indices, :][:, indices]

gm = compute_gm(X, n_components=64, indices=indices, initiliaze_mu=True)

# noprior = ('noprior', None)
# normprior = ('normal', normal_prior(mean, cov))
gmprior = ('gaussian', gm)

goal_joints = ['rfoot','lfoot']

samples = get_motion_samples(selected, 100, 10, sample_rate=2)
sequences = []
for sample in samples:
    goals = []
    for pose in sample:
        goals.append(set_goal(goal_joints, pose))
    sequences.append(goals)

saveframes, plot = True, False

n_epochs, lr, weight_decay, lh_var = 75, 2, 0, 1

# inv_noprior = Inverse_model(noprior, excluded, saveframes=saveframes, plot=plot)
# inv_normal = Inverse_model(normprior, excluded, saveframes=saveframes, plot=plot)
inv_gm = Inverse_model(gmprior, excluded, saveframes=saveframes, plot=plot)

# inv_noprior.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)
# inv_normal.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)

sequence_results = []
for sequence in sequences:
    inv_gm.joints['root'].set_motion(dummy_pose)
    n_epochs_i, lr_i = n_epochs, lr
    for i, goal in enumerate(sequence):
        inv_gm.inverse_kinematics(goal, n_epochs=n_epochs_i, lr=lr_i, lh_var=lh_var, weight_decay=weight_decay)
        sequence_results.append(inv_gm.result)
        if not i:
            n_epochs_i, lr_i = 15, 2
        print(i+1)
    v = Viewer(dummy_joints, sequence_results)
    v.run()

# Frames = [inv_noprior.frames, inv_normal.frames, inv_gm.frames]
#
# for frames in Frames:
#     v = Viewer(dummy_joints, frames)
#     v.run()
