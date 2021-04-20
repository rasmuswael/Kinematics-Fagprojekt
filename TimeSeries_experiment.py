from InverseKinematics import *
from TimeSeries import *
torch.manual_seed(1509)

sample_rate=6

selected = get_fnames(["run", "walk"])
data = parse_selected(selected, sample_rate=sample_rate, limit=15000)
len_array = get_lengths_np(data)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1] - 3)]

dummy_joints, dummy_pose = dummy()

# excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes']
excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist', 'lwrist', 'rclavicle', 'lclavicle']

included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
# mean, cov = compute_parameters_normal(truncate(X))
# mean, cov = mean[indices], cov[indices, :][:, indices]

# means, covs, weights = compute_gm_params(X, n_components=5, indices=indices, covariance_type='full')
n_states=12
model = compute_hmmGauss(X, len_array, n_components=n_states, indices=indices, covariance_type='full')


# noprior = ('noprior', None)
# normprior = ('normal', normal_prior(mean, cov))
# gmprior = ('gaussianmixture', gmm_prior(means, covs, weights))
hmmGaussprior = ('hmmGauss', (gmm_prior(model.means_, model.covars_, torch.zeros(n_states)), model))

goal_joints = ['rfoot', 'lfoot']

samples = get_motion_samples(selected, 50, 2, sample_rate=sample_rate)
sequences = get_goal_sequences(goal_joints, samples, indices)

saveframes, plot = True, False

n_epochs, lr, weight_decay, lh_var = 40, 1, 0, 1e-2
parameters = (n_epochs, lr, weight_decay, lh_var)
# inv_noprior = Inverse_model(noprior, excluded, saveframes=saveframes, plot=plot)
# inv_normal = Inverse_model(normprior, excluded, saveframes=saveframes, plot=plot)
# inv_gm = Inverse_model(gmprior, indices, saveframes=saveframes, plot=plot)
inv_hmmGauss = Inverse_model(hmmGaussprior, indices, saveframes=saveframes, plot=plot)

gen_timeseries(inv_hmmGauss, sequences, parameters, samples, view=True, init_pose='first pose', interpolate=True, sample_rate=sample_rate)

# for j, sequence in enumerate(sequences):
#     init_pose = samples[j][0]
#     for key, value in init_pose.items():
#         init_pose[key] = torch.tensor(value)
#     inv_hmmGauss.pose = init_pose
#     sequence_results = []
#     n_epochs_i, lr_i = n_epochs, lr
#     for i, goal in enumerate(sequence):
#         inv_hmmGauss.inverse_kinematics(goal, n_epochs=n_epochs_i, lr=lr_i, lh_var=lh_var, weight_decay=weight_decay)
#         sequence_results.append(inv_hmmGauss.pose)
#         if not i:
#             n_epochs_i, lr_i = 40 , 1e-1
#         print(i+1)
#     v = Viewer(dummy_joints, sequence_results, trajectories=[unpack_sequence(goal_joint, sequence) for goal_joint in goal_joints])
#     v.run()




# for j, sequence in enumerate(sequences):
#     inv_gm.pose = dummy_pose
#     sequence_results = []
#     n_epochs_i, lr_i = n_epochs, lr
#     for i, goal in enumerate(sequence):
#         inv_gm.inverse_kinematics(goal, n_epochs=n_epochs_i, lr=lr_i, lh_var=lh_var, weight_decay=weight_decay)
#         sequence_results.append(inv_gm.pose)
#         if not i:
#             n_epochs_i, lr_i = 15, 1e-1
#         print(i+1)
#     v = Viewer(dummy_joints, sequence_results, trajectories=[unpack_sequence(goal_joint, sequence) for goal_joint in goal_joints])
#     v.run()

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
