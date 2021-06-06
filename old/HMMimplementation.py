from compute_models import *
# from pyTorch_3Dviewer import *
from dummy import dummy

selected = get_fnames(["walk"])
data = parse_selected(selected, sample_rate=2, limit=5000)
len_array = get_lengths_np(data)
X, y = gather_all_np(data)
X = X[:, :(X.shape[1])-3]

excluded = ['root', 'lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'rfoot',
            'lfoot', 'head', 'rwrist', 'lwrist', 'rclavicle', 'lclavicle']
included, indices = exclude(excluded, return_indices=True)
X = remove_excluded(truncate(X), indices, type = 'numpy')

# model = hmm.GMMHMM(n_components=16, n_mix=5, covariance_type="full")
model = hmm.GaussianHMM(n_components=3, covariance_type="full")

model = model.fit(X, lengths=len_array)

for mean in model.means_:
    draw_cluster(mean, indices, type='numpy')
sample,states = model.sample(200)
print(states)
motions = array2motions(torch.tensor(sample), indices)

dummy_joints, dummy_pose = dummy()
viewer = Viewer(dummy_joints, motions)
viewer.run()


