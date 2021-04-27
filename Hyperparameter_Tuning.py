from compute_models import *
import matplotlib.pyplot as plt
sample_rate = 6

# Lets create a cross validation loop for the HMM Gaussian model
selected = get_fnames(["walk", "run"])
excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist', 'lwrist', 'rclavicle', 'lclavicle']


included, indices = exclude(excluded, return_indices=True, root_exclude=[1])
data = parse_selected(selected, sample_rate=6, limit=50000)
K = 5
folds = get_fold(data, K, type='Kfold', group='motion')
X, y, len_arrays = get_Kfold_mat(data, folds)
X = X[:, :(X.shape[1] - 3)]
N = X.shape[0]

n_states_candidates = np.arange(2,20)
results = [0] * len(n_states_candidates)
for i in range(K):
    print(i)
    Xtrain, Xtest, len_array_train, len_array_test = get_train_test_split(X, i, len_arrays)
    for j, n_states in enumerate(n_states_candidates):
        print(j)
        model = compute_hmmGauss(Xtrain, len_array_train, n_states, indices)
        results[j] += model.score(remove_excluded(truncate(Xtest), indices, type='numpy'), lengths=len_array_test) * sum(len_array_test) / N
    print(results)

print(results)
plt.plot(n_states_candidates, results)
plt.scatter(n_states_candidates, results)
plt.show()