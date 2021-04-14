from InverseKinematics import *
torch.manual_seed(1509)

selected = get_fnames(["walk"])
data = parse_selected(selected, sample_rate=2, limit=20000)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1] - 3)]

dummy_joints, dummy_pose = dummy()

# excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes']
excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist', 'lwrist', 'rclavicle', 'lclavicle']

included, indices = exclude(excluded, return_indices=True, root_exclude=[0, 1])
# mean, cov = compute_parameters_normal(truncate(X))
# mean, cov = mean[indices], cov[indices, :][:, indices]

means, covs, weights = compute_gm_params(X, n_components=5, indices=indices)

# noprior = ('noprior', None)
# normprior = ('normal', normal_prior(mean, cov))
gmprior = ('gaussian', gmm_prior(means, covs, weights))

goal_joints = ['rfoot', 'lfoot']

samples = get_motion_samples(selected, 100, 2, sample_rate=2)
sequences = get_goal_sequences(goal_joints, samples, indices)

saveframes, plot = True, False

n_epochs, lr, weight_decay, lh_var = 75, 6, 0, 1

# inv_noprior = Inverse_model(noprior, excluded, saveframes=saveframes, plot=plot)
# inv_normal = Inverse_model(normprior, excluded, saveframes=saveframes, plot=plot)
inv_gm = Inverse_model(gmprior, indices, saveframes=saveframes, plot=plot)

# inv_noprior.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)
# inv_normal.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)

for j, sequence in enumerate(sequences):
    inv_gm.pose = dummy_pose
    sequence_results = []
    n_epochs_i, lr_i = n_epochs, lr
    for i, goal in enumerate(sequence):
        inv_gm.inverse_kinematics(goal, n_epochs=n_epochs_i, lr=lr_i, lh_var=lh_var, weight_decay=weight_decay)
        sequence_results.append(inv_gm.pose)
        if not i:
            n_epochs_i, lr_i = 15, 1e-1
        print(i+1)
    v = Viewer(dummy_joints, sequence_results, trajectories=[unpack_sequence(goal_joint, sequence) for goal_joint in goal_joints])
    v.run()

# for j, sequence in enumerate(sequences):
#     inv_normal.pose = dummy_pose
#     sequence_results = []
#     n_epochs_i, lr_i = n_epochs, lr
#     for i, goal in enumerate(sequence):
#         inv_normal.inverse_kinematics(goal, n_epochs=n_epochs_i, lr=lr_i, lh_var=lh_var, weight_decay=weight_decay)
#         sequence_results.append(inv_normal.pose)
#         if not i:
#             n_epochs_i, lr_i = 5, 1
#         print(i+1)
#     v = Viewer(dummy_joints, sequence_results, points=unpack_sequence(goal_joints[0], sequence))
#     v.run()

# Frames = [inv_noprior.frames, inv_normal.frames, inv_gm.frames]
#
# for frames in Frames:
#     v = Viewer(dummy_joints, frames)
#     v.run()
