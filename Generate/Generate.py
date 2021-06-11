from inverse_kinematics.quick_setup import *

seed = 4256

type = 'hmmGauss'
### Hent modelparametre her
model_fname = "hmmGauss_16hstates_sr6"
path = f"./models/{model_fname}"
parameters = load_param(path, type)

prior = get_prior(type, parameters)

excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist',
                    'lwrist', 'rclavicle', 'lclavicle']
included, indices = exclude(excluded, return_indices=True, root_exclude=[1])

trainselected, testselected = get_trainandtestsample_names(seed)

goal_joints = ['rfoot', 'lfoot']
sample_rate = 6
testsamples, sampleids = get_motion_samples(testselected, length=None, random=False, sample_rate=sample_rate)
sequences, trunc_samples = get_goal_sequences(goal_joints, testsamples, indices, return_trunc_samples=True)

saveframes, plot = True, False

n_epochs, lr, weight_decay, lh_var = 250, 1, 0, 1e-4
parameters = (n_epochs, lr, weight_decay, lh_var)

inv_model = Inverse_model(prior, indices, saveframes=saveframes, plot=plot)


gen_timeseries(inv_model, sequences, parameters, trunc_samples, view=True, init_pose='model pose', opt_pose='self',
               interpolate=True, sample_rate=sample_rate, saveids=sampleids)