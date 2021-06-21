from inverse_kinematics.quick_setup import *

### SCRIPT TO COUNT THE TOTAL AMOUNT OF DATAPOINTS IN EACH SUBCATEGORY ###

seed = 4256

type = 'normal'
trainselected, testselected = get_trainandtestsample_names(seed)
print(len(trainselected), len(testselected))
ntrainsequences = sum([len(sublist) for sublist in trainselected.values()])
ntestsequences = sum([len(sublist) for sublist in testselected.values()])
ntrainwalkseq = sum([sequence[1] == 'walk' for sublist in trainselected.values() for sequence in sublist])
ntrainrunseq = sum([sequence[1] == 'run' for sublist in trainselected.values() for sequence in sublist])
ntestwalkseq = sum([sequence[1] == 'walk' for sublist in testselected.values() for sequence in sublist])
ntestrunseq = sum([sequence[1] == 'run' for sublist in testselected.values() for sequence in sublist])
print(ntrainsequences, ntestsequences, ntrainwalkseq, ntrainrunseq, ntestwalkseq, ntestrunseq)
print(testselected)
data_pool, sample_rate, limit = 'manual', 6, None
Xytrain, len_array, indices = quick_setup(None, data_pool=trainselected, sample_rate=sample_rate, limit=None, excluded=[], returny=True)
Xytest, len_array, indices = quick_setup(None, data_pool=testselected, sample_rate=sample_rate, limit=None, excluded=[], returny=True)
Xtrain, ytrain = Xytrain
Xtest, ytest = Xytest
print(f"Amount of training frames (subsampled to 20 fps): {len(ytrain)}\nAmount of training sequences: {ntrainsequences}\nAmount of 'walk' training frames (subsampled to 20 fps): {np.sum(ytrain=='walk')}\nAmount of 'walk' training sequences: {ntrainwalkseq}\nAmount of 'run' training frames (subsampled to 20 fps): {np.sum(ytrain=='run')}\nAmount of 'run' training sequences:{ntrainrunseq}")
print(len(indices))
