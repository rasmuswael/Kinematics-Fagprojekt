from compute_models import *
# from pyTorch_3Dviewer import *
from sklearn.neighbors import LocalOutlierFactor

joints, dummy_pose = dummy()
selected = get_fnames(["walk"])
data = parse_selected(selected, sample_rate=2, limit=25000)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1])-3]

for i, x in enumerate(truncate(X, root_limit=np.array([[-179., 179.], [0., 0.],[-179., 179.]]))[:,:3]):
    if np.any(abs(x)>90):
        print(i)

# excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'rfoot', 'lfoot', 'head',
#             'rwrist', 'lwrist', 'rclavicle', 'lclavicle']
# included, indices = exclude(excluded, return_indices=True)
# X = remove_excluded(truncate(X), indices, type = 'numpy')
#
#

# 
# X1 = np.append(X,np.zeros((1,X.shape[1])),axis = 0)
# X2 = np.insert(X,0,np.zeros((1,X.shape[1])),axis = 0)
# diff = X2-X1
# #diff = diff[1:1500]
# # sorted = np.sort(X1-X2, axis = 1)
# lof = LocalOutlierFactor(novelty=True)
# lof.fit(diff)
# outlier = lof.negative_outlier_factor_
# 
# curr_frame = 0
# outlier_actions = []
# for subject in data.keys():
#     for i, action_length in enumerate(data[subject]['lengths']):
#         if np.any(outlier[curr_frame+1:action_length] < -8):
#             outlier_actions.append(data[subject]['actionid'][i])
#             curr_frame += action_length
# 
# print(outlier_actions)










# motions = array2motions(truncate(X)[890:1000,:])
# v = Viewer(joints, motions)
# v.run()
