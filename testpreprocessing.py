from compute_models import *
from pyTorch_3Dviewer import *

joints, dummy_pose = dummy()
selected = get_fnames(["walk"])
data = parse_selected(selected, sample_rate=2, limit=20000)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1])-3]

# for i, x in enumerate(truncate(X, root_limit=np.array([[-179., 179.], [0., 0.],[-179., 179.]]))[:,:3]):
#     if np.any(abs(x)>90):
#         print(i)
#
# excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'rfoot', 'lfoot', 'head',
#             'rwrist', 'lwrist', 'rclavicle', 'lclavicle']
# included, indices = exclude(excluded, return_indices=True, root_exclude=[0, 1])
# X = remove_excluded(truncate(X), indices, type = 'numpy')

motions = array2motions(truncate(X)[890:1000,:])
v = Viewer(joints, motions)
v.run()
