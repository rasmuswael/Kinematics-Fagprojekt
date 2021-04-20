from scipy import interpolate
from compute_models import *
from pyTorch_3Dviewer import *

def interpolatearray(X, sample_rate):
    num_frames, num_features = X.shape
    x = np.arange(num_frames) * sample_rate
    x_interpolated = np.arange(0, x[-1])
    f = interpolate.interp1d(x, X[:, 0])
    X_interpolated = f(x_interpolated).reshape(-1,1)
    for i in range(num_features - 1):
        f = interpolate.interp1d(x, X[:, i + 1])
        X_interpolated = np.hstack((X_interpolated, f(x_interpolated).reshape(-1,1)))
    return X_interpolated

def gen_timeseries(inv_model, sequences, parameters, samples, view=True, init_pose='first pose', interpolate=False, sample_rate=1):
    dummy_joints, dummy_pose = dummy()
    for j, sequence in enumerate(sequences):
        if init_pose == 'dummy':
            init_pose = dummy_pose
        elif init_pose == 'first pose':
            init_pose = samples[j][0]
            for key, value in init_pose.items():
                init_pose[key] = torch.tensor(value)

        inv_model.pose = init_pose
        sequence_results = []
        n_epochs, lr, weight_decay, lh_var = parameters
        for i, goal in enumerate(sequence):
            inv_model.inverse_kinematics(goal, n_epochs=n_epochs, lr=lr, lh_var=lh_var,
                                            weight_decay=weight_decay)
            sequence_results.append(inv_model.pose)
            print(i + 1)
        if interpolate:
            sequence_results = array2motions(interpolatearray(motions2array(sequence_results), sample_rate))

        if view:
            # v = Viewer(dummy_joints, sequence_results,
            #            trajectories=[unpack_sequence(goal_joint, sequence) for goal_joint in goal_joints])
            v = Viewer(dummy_joints, sequence_results)
            v.run()
