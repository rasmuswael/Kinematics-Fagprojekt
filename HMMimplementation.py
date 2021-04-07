from hmmlearn import hmm
import numpy as np
from get_subjects import *
from compute_models import *
from pyTorch_3Dviewer import *
from dummy import dummy
import matplotlib.pyplot as plt


selected = get_fnames(["walk"])
data = parse_selected(selected, relative_sample_rate=1, limit=20000)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1])-3]

included, indices = exclude(return_indices=True)
X = remove_excluded(truncate(X), indices, type = 'numpy')


model = hmm.GMMHMM(n_components=4, n_mix=15, covariance_type="spherical", n_iter=200)

model = model.fit(X)

sample,states = model.sample(1000)

motions = []
for x in sample:
    pose = array2pose(torch.tensor(x), indices)
    motions.append(pose)

print(states)
dummy_joints, dummy_pose = dummy()
viewer = Viewer(dummy_joints, motions)
viewer.run()


