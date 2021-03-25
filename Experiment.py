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

prior = normal_prior(mean, cov)

goal_joints = ['rfoot']
pose = {'rfemur': [40, 0, 0]}

# goal_joints = ['rfoot', 'rhand']
# pose = {'rfemur': [40, 0, 0], 'rhumerus': [-90,0,0]}

goal = set_goal(goal_joints, pose)

frames = inverse_kinematics(goal, ('normal',prior), excluded, saveframes=True)
v = Viewer(dummy_joints, frames)
v.run()
# inverse_kinematics(goal)