from inverse_kinematics.compute_models import *
import numpy as np
import pandas as pd
from get_subjects import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
columns_excl = [
           'lowerback x','lowerback y','lowerback z',
           'upperback x','upperback y','upperback z',
           'thorax x','thorax y','thorax z',
           'lowerneck x','lowerneck y','lowerneck z',
           'upperneck x','upperneck y','upperneck z',
           'rhumerus x','rhumerus y','rhumerus z',
           'rradius x',
           'lhumerus x','lhumerus y','lhumerus z',
           'lradius x',
           'rfemur x', 'rfemur y', 'rfemur z',
           'rtibia x',
           'rfoot x', 'rfoot z',
           'lfemur x', 'lemur y', 'lfemur z',
           'ltibia x',
           'lfoot x', 'lfoot z']


data = pd.read_pickle("./run_data_trun.pkl")
data = data.loc[('104_48')]


x = data.loc[:, columns_excl].values
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

ax.scatter(principalDf.loc[:, 'principal component 1']
               , principalDf.loc[:, 'principal component 2']
               , s = 50)
ax.grid()
plt.show()










