import pandas as pd
from compute_models import *

selected = get_fnames(["run"])
data = parse_selected(selected, sample_rate=1, limit=None)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1]) - 3]


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
           'lfemur x', 'lemur y', 'lfemur z',
           'ltibia x',
           'lfoot x', 'lfoot z',
           'ltoes x',
           ]
# for i in data["02"]['actions'][0].keys():
#     columns.append(i)


dataframe = {}
actionids = []
lengths = []

for subject in data.keys():
    actionids.extend(data[subject]['actionid'])
    lengths.extend(data[subject]['lengths'])

curr_frame = 0
for i, length in enumerate(lengths):
    dataframe[actionids[i]] = pd.DataFrame(X[curr_frame : curr_frame + length], columns = columns)
    curr_frame += length

df = pd.concat(dataframe, keys = actionids)
df.to_pickle('./run_data.pkl')

#How to get just one action example
#df.loc[('02_01')]


