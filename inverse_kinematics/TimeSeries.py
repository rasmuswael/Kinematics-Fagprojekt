from scipy import interpolate
from inverse_kinematics.InverseKinematics import *

def interpolatearray(X, sample_rate, output_fps=120):
    input_fps = 120
    fps_ratio = input_fps // output_fps
    num_frames, num_features = X.shape
    x = np.arange(num_frames) * (sample_rate // fps_ratio)
    x_interpolated = np.arange(0, x[-1])
    f = interpolate.interp1d(x, X[:, 0])
    X_interpolated = f(x_interpolated).reshape(-1,1)
    for i in range(num_features - 1):
        f = interpolate.interp1d(x, X[:, i + 1])
        X_interpolated = np.hstack((X_interpolated, f(x_interpolated).reshape(-1,1)))
    return X_interpolated

def gen_timeseries(inv_model, sequences, parameters, samples, view=True, init_pose='first pose', opt_pose='self',
                   interpolate=False, sample_rate=1, output_fps=120, save=False):
    dummy_joints, dummy_pose = dummy()
    n_epochs, lr, weight_decay, lh_var = parameters
    for j, sequence in enumerate(sequences):
        if init_pose == 'dummy':
            init_pose = dummy_pose
        elif init_pose == 'first pose':
            init_pose = samples[j][0]
            for key, value in init_pose.items():
                init_pose[key] = torch.tensor(value)
        elif init_pose == 'model pose':
            if inv_model.update:
                inv_model.update = False
                inv_model.inverse_kinematics(sequence[0], n_epochs=n_epochs, lr=lr, lh_var=lh_var,
                                                weight_decay=weight_decay, opt_pose=opt_pose)
                inv_model.update = True
            else:
                inv_model.inverse_kinematics(sequence[0], n_epochs=n_epochs, lr=lr, lh_var=lh_var,
                                                         weight_decay=weight_decay, opt_pose=opt_pose)
            init_pose = inv_model.pose
        inv_model.pose = init_pose
        sequence_results = []
        hstate_probs = []
        for i, goal in tqdm(enumerate(sequence)):
            inv_model.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var,
                                            weight_decay=weight_decay, opt_pose=opt_pose)
            sequence_results.append(inv_model.pose)
            if inv_model.update:
                hstate_probs.append(inv_model.hstates.detach().numpy())

        goal_joints = goal.keys()
        if interpolate:
            sequence_results = array2motions(interpolatearray(motions2array(sequence_results), sample_rate=sample_rate,
                                                              output_fps=output_fps), type='numpy')

        if view:
            v = Viewer(dummy_joints_np(), sequence_results, trajectories=[unpack_sequence(goal_joint, sequence) for goal_joint in goal_joints],
                       sample_rate=sample_rate, hstate_probs=hstate_probs, fps=output_fps)
            v.run()



