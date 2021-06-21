from scipy import interpolate
from inverse_kinematics.InverseKinematics import *
from inverse_kinematics.amc_creator import amc_converter
import pickle

def interpolatearray(X, sample_rate, output_fps=120):
    """Function for linearly interpolating between generated frames
    :arg X, numpy array to be interpolated
    :arg sample_rate, the sub-sampling rate
    :arg output_fps, output fps of the interpolation. Defaults to 120 fps"""
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
                   interpolate=False, sample_rate=1, output_fps=120, saveids=[]):
    """The main function handling the time series generation loop. Loops over all goal point in a sequence and
    performs inverse kinematics using inv_model.
    :arg inv_model, see InverseKinematics.py
    :arg sequences, list of goal point sequences
    :arg parameters, tuple (n_epochs, lr, weight_decay, lh_var) containing the parameters for the inverse_kinematics solver.
    :arg samples, list of motions samples to get the goal sequences
    :arg view, view result
    :arg init_pose, initial pose for the model. Either 'dummy' meaning t-pose, 'first pose' meaning the first pose in the
    motion sample or 'model pose' meaning that the initial pose is computed by the model
    :arg opt_pose, whether the optimisation starts from the previous pose, 'self' or the dummy pose, 'dummy'
    :arg interpolate, whether the motion should be interpolated.
    :arg sample_rate, the rate at which the motion was sampled
    :arg output_fps, output fps of the motion. Only relevant if interpolate=True (see interpolatearray)
    :arg saveids, ids for the generated motions. If empty, motions are NOT saved"""
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
        nframes = len(sequence)
        for i, goal in enumerate(sequence):
            print(f'Frame: {i} / {nframes}')
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

        if len(saveids):
            amc_converter(sequence_results, f"{inv_model.prior[0]}-{saveids[j]}")
            if inv_model.update:
                with open(f"./hstate_probs_results/hstate_probs_{inv_model.prior[0]}-{saveids[j]}.txt", "wb") as fp:  # Pickling
                    pickle.dump(hstate_probs, fp)

