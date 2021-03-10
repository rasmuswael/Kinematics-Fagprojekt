import numpy as np
fname = r'../models/normal_params.npz'
npzfile = np.load(fname, allow_pickle=True)
means = npzfile['arr_0']
covs = npzfile['arr_1']
queries = npzfile['arr_2']
