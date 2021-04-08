from compute_models import *
from pyTorch_3Dviewer import *

joints, dummy_pose = dummy()
selected = get_fnames(["walk"])
data = parse_selected(selected, relative_sample_rate=2, limit=500)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1])-3]

excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'rfoot', 'lfoot']
included, indices = exclude(excluded, return_indices=True)
X = remove_excluded(truncate(X), indices, type = 'numpy')

motions = array2motions(X, indices)
v = Viewer(joints, motions)
v.run()
