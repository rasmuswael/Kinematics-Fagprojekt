import matplotlib.pyplot as plt

from inverse_kinematics.quick_setup import *
import seaborn as sns
import pandas as pd

### SCRIPT TO COMPUTE THE LOSS WHEN INTERPOLATING ###


data_pool = 'manual'

excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist',
                    'lwrist', 'rclavicle', 'lclavicle']
included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
columns = ['root x', 'root y', 'root z',
           'lowerback x','lowerback y','lowerback z',
           'upperback x','upperback y','upperback z',
           'thorax x','thorax y','thorax z',
           'lowerneck x','lowerneck y','lowerneck z',
           'upperneck x','upperneck y','upperneck z',
           'head x','head y','head z',
           'rclavicle y','rclavicle z',
           'rhumerus x','rhumerus y','rhumerus z',
           'rradius x',
           'rwrist y',
           'rhand x', 'rhand z',
           'rfingers x',
           'rthumb x','rthumb z',
           'lclavicle y','lclavicle z',
           'lhumerus x','lhumerus y','lhumerus z',
           'lradius x',
           'lwrist y',
           'lhand x','lhand z',
           'lfingers x',
           'lthumb x',  'lthumb z',
           'rfemur x', 'rfemur y', 'rfemur z',
           'rtibia x',
           'rfoot x', 'rfoot z',
           'rtoes x',
           'lfemur x', 'lfemur y', 'lfemur z',
           'ltibia x',
           'lfoot x', 'lfoot z',
           'ltoes x',
           ]
columns = np.array(columns)[indices]
print(columns, len(columns))
sort_ind = np.argsort([ang[1:] for ang in columns])
path = f'./data_asarray/{data_pool}'
# np.savez(path, X, y)
# print('done')
arrays = np.load(f'{path}.npz')
X, y = arrays['arr_0'], arrays['arr_1']

sample_rate =  6
Xsub = X[::sample_rate]
Xsub = interpolatearray(Xsub, sample_rate, output_fps=120)
X = X[:Xsub.shape[0]-X.shape[0]]
print(f'Check: {X.shape}, {Xsub.shape}')
Xloss = np.abs(X - Xsub)
Xloss = Xloss[~np.all(Xloss == 0., axis=1)]
Xloss = pd.DataFrame(Xloss[:,sort_ind], columns=columns[sort_ind])

print(Xloss.mean(axis=0))
sns.pointplot(data=Xloss, join=False, markers='.', scale=0.4, errwidth=0.8, capsize=0.25)
plt.title('Subsampling interpolation loss - 120 FPS to 20 FPS', fontweight='bold')
plt.ylabel('Absolute angle difference')
plt.xticks(rotation=45)
plt.show()
