import numpy as np

from inverse_kinematics.quick_setup import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

### SCRIPT TO GENERATE PLOTS ###
data_pool, sample_rate, limit = 'manual', 6, None

excluded = ['lfingers', 'lthumb', 'ltoes', 'rfingers', 'rthumb', 'rtoes', 'rhand', 'lhand', 'head', 'rwrist',
                    'lwrist', 'rclavicle', 'lclavicle']

included, indices = exclude(excluded, return_indices=True, root_exclude=[1])

normalpath = "./models/normal_sr6"
normalparams = load_param(normalpath, "normal")
hmmGausspath = "./models/hmmGauss_16hstates_sr6"
hmmGaussmodel = load_param(hmmGausspath, "hmmGauss")

Gauss = normal_prior(normalparams[0], normalparams[1])
hmmGauss = gmm_prior(hmmGaussmodel.means_, hmmGaussmodel.covars_, torch.tensor([1/len(hmmGaussmodel.means_)]*len(hmmGaussmodel.means_)))
n_states = hmmGauss.n_components

path = f'./data_asarray/{data_pool}'
# np.savez(path, X, y)
# print('done')
arrays = np.load(f'{path}.npz')
X, y = arrays['arr_0'], arrays['arr_1']

# indices = np.argsort(y)
types, counts = np.unique(y, return_counts=True)



N = 10000

GaussX = Gauss.rsample(torch.Size([N])).numpy()
# X = X[indices, :]
standard = StandardScaler().fit(X)
X = standard.transform(X)
GaussX = standard.transform(GaussX)
n_components = 4
pca = PCA(n_components=n_components)
pca.fit(X)
print(f'Explained variance of PC components 1 and 2:\nPC 1: {round(pca.explained_variance_ratio_[0]*100,1)}%\nPC 2: {round(pca.explained_variance_ratio_[1]*100,1)}%')

pcdf = pd.DataFrame(data = pca.transform(X), columns = [f'PC {i + 1}' for i in range(n_components)])
Gausspcdf = pd.DataFrame(data = pca.transform(GaussX), columns = [f'PC {i + 1}' for i in range(n_components)])
# hmmGausspcdf = pd.DataFrame(data = pca.transform(hmmGaussX), columns = [f'PC {i + 1}' for i in range(n_components)])
pcdf['type'] = y
# sns.set_theme(style="ticks")
# sns.pairplot(pcdf, hue="type",corner=True, markers='+',height=1.2)
# plt.show()

print(hmmGaussmodel.startprob_)
sns.heatmap(hmmGaussmodel.transmat_, linewidth=0.5, annot=np.round(hmmGaussmodel.transmat_, 2), cmap='Blues')
plt.title('Transition matrix for the Gaussian HMM model', fontweight='bold')
plt.show()

x, y = 'PC 1', 'PC 2'

xlim = (-8,8)
ylim = (-8,8)

cmap = 'flare_r'
background = sns.color_palette(cmap, as_cmap=True)(0)

fig, ax = plt.subplots(1,3, sharex=True, sharey=True)
fig.suptitle('Density plots for pose data', fontweight='bold')
plt.setp(ax, xlim=xlim, ylim=ylim)
sns.kdeplot(
    ax=ax[0], data=pcdf[pcdf['type'] == 'walk'], x=x, y=y,
    fill=True, thresh=0, levels=50, cmap=cmap
)
ax[0].set_title('Distribution of walking poses')
ax[0].set_facecolor(background)
sns.kdeplot(
    ax=ax[1], data=pcdf[pcdf['type'] == 'run'], x=x, y=y,
    fill=True, thresh=0, levels=50, cmap=cmap
)
ax[1].set_title('Distribution of running poses')
ax[1].set_facecolor(background)
sns.kdeplot(
    ax=ax[2], data=Gausspcdf, x=x, y=y,
    fill=True, thresh=0, levels=50, cmap=cmap, cbar=True, cbar_kws={'ticks': [0.0, 0.2, 0.4, 0.5, 0.7, 0.8, 1.0]}
)
ax[2].set_title('Gaussian distribution over poses')
ax[2].set_facecolor(background)
plt.show()


n_rows = 2
n_columns = n_states // n_rows
fig = plt.figure(constrained_layout=True, figsize=(10, 4))
subfigs = fig.subfigures(n_rows, n_columns, wspace=0.07)
fig.suptitle('Mean pose for Gaussian HMM emissions', fontweight='bold')
for i in tqdm(range(n_states)):
    mean_pose = hmmGauss.means[i]
    axis = (i//n_columns, i%n_columns)
    draw_cluster(mean_pose, indices, fig=subfigs[axis[0], axis[1]])
    subfigs[axis[0], axis[1]].suptitle(f'State {i+1}')
plt.show()


fig, ax = plt.subplots(n_rows, n_columns, sharex=True, sharey=True)
fig.suptitle('Density plots for Gaussian HMM emissions', fontweight='bold')
plt.setp(ax, xlim=xlim, ylim=ylim)
for i in tqdm(range(n_states)):
    weights = torch.zeros(n_states)
    weights[i] = 1
    hmmGauss.weights = weights
    hmmGaussX = standard.transform(hmmGauss.sample(N).numpy())
    hmmGausspcdf = pd.DataFrame(data=pca.transform(hmmGaussX), columns=[f'PC {i + 1}' for i in range(n_components)])
    axis = (i//n_columns, i%n_columns)
    sns.kdeplot(
        ax=ax[axis[0], axis[1]], data=hmmGausspcdf, x=x, y=y,
        fill=True, thresh=0, levels=50, cmap=cmap
    )
    ax[axis[0], axis[1]].set_title(f'State {i+1}')
    ax[axis[0], axis[1]].set_facecolor(background)
plt.show()


