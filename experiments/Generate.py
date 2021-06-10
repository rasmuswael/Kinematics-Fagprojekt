from inverse_kinematics.quick_setup import *


type = 'hmmGauss'
### Hent modelparametre her
model_fname = "hmmGauss_experiment1"
path = f"./models/{model_fname}"
parameters =

### Eller træn en ny her
X, len_array, indices = quick_setup(selected, data_pool="manual", sample_rate=1, limit=None, excluded=[])
parameters = train_prior(X, indices, type, )

prior = get_prior(type)
goal_joints = ['rfoot', 'lfoot']

examples = selected
# examples = {'104': [('104_56','')]}

np.random.seed(seed)
samples = get_motion_samples(examples, 100, 2, sample_rate=sample_rate)
sequences, trunc_samples = get_goal_sequences(goal_joints, samples, indices, return_trunc_samples=True)

saveframes, plot = True, False

n_epochs, lr, weight_decay, lh_var = 100, 1e-1 * sample_rate, 0, 1e-2
parameters = (n_epochs, lr, weight_decay, lh_var)

# inv_noprior = Inverse_model(noprior, excluded, saveframes=saveframes, plot=plot)
# inv_normal = Inverse_model(normprior, excluded, saveframes=saveframes, plot=plot)
# inv_gm = Inverse_model(gmprior, indices, saveframes=saveframes, plot=plot)
inv_hmmGauss = Inverse_model(hmmGaussprior, indices, saveframes=saveframes, plot=plot)


gen_timeseries(inv_hmmGauss, sequences, parameters, trunc_samples, view=True, init_pose='first pose', opt_pose='self',
               interpolate=True, sample_rate=sample_rate)