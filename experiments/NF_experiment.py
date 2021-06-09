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

steps, lr = 1000, 5e-3

nfprior = ('normalizingflows', nf_prior(compute_NF(X, steps=steps, indices=indices, lr=lr)))

# goal_joints = ['rfoot']
# pose = {'rfemur': [40, 0, 0]}

goal_joints = ['rfoot', 'lfoot']
pose = {'rfemur': [25, 0, 0], 'lfemur': [-25, 0, 0]}

goal = set_goal(goal_joints, pose)

saveframes, plot = True, True

n_epochs, lr, weight_decay, lh_var = 500, 1, 0, 1

inv_nf = Inverse_model(nfprior, indices, saveframes=saveframes, plot=plot)

inv_nf.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var, weight_decay=weight_decay)

v = Viewer(dummy_joints_np(), inv_nf.frames)
v.run()
