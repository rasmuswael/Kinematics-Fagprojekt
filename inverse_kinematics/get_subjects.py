import os
import numpy as np
from inverse_kinematics.pyTorch_parser import parse_asf, parse_amc
from inverse_kinematics.amc_parser import parse_asf_np, parse_amc_np
from tqdm import tqdm
import random


os.chdir(os.path.dirname(os.path.abspath(__file__)))
def get_subjects():
    dir_path = "./data/subjects"

    subjects = {}
    for filename in os.listdir(dir_path):
        subject_path = f"{dir_path}/{filename}"
        subject_number = ''.join(map(str, [int(s) for s in filename.split()[0] if s.isdigit()]))
        if not len(subject_number) - 1:
            subject_number = ''.join(map(str, [0, subject_number]))
        s = open(subject_path, "r")

        info = {}
        for i, line in enumerate(s):
            if i not in [0, 1, 2]:
                key = line.split("\t")[0]
                if key != "\n":
                    keywords = line.split("\t")[-2]
                    if key not in keywords:
                        info[key] = keywords
                    else:
                        info[key] = "None"
        subjects[subject_number] = info
    subjects = {k: v for k, v in sorted(subjects.items(), key=lambda item: int(item[0]))}
    return subjects

def get_fnames(queries, limit=0, subjects=get_subjects()):
    selected = {}
    for subject_number, file in subjects.items():
        match = []
        for filename, description in file.items():
            matching_queries = []
            for query in queries:
                if limit and query == description:
                    matching_queries.append(query)
                elif query in description:
                    matching_queries.append(query)
            if matching_queries:
                match.append((filename, " and ".join(matching_queries)))
        if match:
            selected[subject_number] = match
    return selected

def get_manual_names(types):
    """The sequel"""
    selected = {}
    for type in types:
        dir = f"./{type}_data"
        for filename in os.listdir(dir):
            fname = str(filename)[:-4]
            subject = fname[:-3]
            if subject not in selected:
                selected[subject] = [(fname, type)]
            else:
                selected[subject].append((fname, type))
    return selected

def get_trainandtestsample_names(seed):
    np.random.seed(seed)
    trainselected, testselected = {}, {}
    types = ('run', 'walk')
    for type in types:
        dir = f"./{type}_data"
        samples = []
        data = sorted(os.listdir(dir))
        samples.extend(np.random.choice(data, 5, replace=False))
        for filename in samples:
            fname = str(filename)[:-4]
            subject = fname[:-3]
            if subject not in testselected:
                testselected[subject] = [(fname, type)]
            else:
                testselected[subject].append((fname, type))
        for filename in data:
            if filename not in samples:
                fname = str(filename)[:-4]
                subject = fname[:-3]
                if subject not in trainselected:
                    trainselected[subject] = [(fname, type)]
                else:
                    trainselected[subject].append((fname, type))
    return trainselected, testselected


def parse_selected(selected, type="numpy", sample_rate=1, limit=None, ignore_translation=True, sep_trans_root=True, motions_as_mat=True):
    count = 0
    done = False
    data = {}
    for key, file in tqdm(selected.items()):
        asf_path = f'./data/{key}/{key}.asf'
        if type == "numpy":
            joints = parse_asf_np(asf_path)
        else:
            joints = parse_asf(asf_path)

        actions, labels, lengths, actionid = [], [], [], []
        for filename, label in file:
            amc_path = f'./data/{key}/{filename}.amc'
            if type == "numpy":
                motions = parse_amc_np(amc_path, ignore_translation)
            else:
                motions = parse_amc(amc_path, ignore_translation)

            sampled_motions = []
            for i, motion in enumerate(motions):
                if not i % sample_rate:
                    if sep_trans_root:
                        motions[i]['translation'] = motion['root'][:3]
                        motions[i]['root'] = motion['root'][3:]
                    sampled_motions.append(motions[i])
            motions = sampled_motions
            if limit:
                count += len(motions)
                if count >= limit:
                    motions = motions[:(len(motions) - (count - limit))]
                    done = True
            length = len(motions)
            if motions_as_mat:
                first = 1
                motions_np = {}
                for motion in motions:
                    for joint in motions[0].keys():
                        if first:
                            motions_np[joint] = np.array(motion.copy()[joint])
                        else:
                            motions_np[joint] = np.vstack((motions_np[joint], np.array(motion[joint])))
                    if first:
                        first = 0
                motions = motions_np
            actions.append(motions)
            labels.append(label)
            lengths.append(length)
            actionid.append(filename)
            if done:
                break
        data[key] = {}
        data[key]['joints'] = joints
        data[key]['actions'] = actions
        data[key]['labels'] = labels
        data[key]['lengths'] = lengths
        data[key]['actionid'] = actionid

        if done:
            return data
    return data


def gather_all_np(data ,big_matrix=True):
    """Requires data as actions as np matrix. See parse_selected"""
    first = 1
    for set in tqdm(data.values()):
        labels = set['labels']
        for i, action in enumerate(set['actions']):
            if first:
                gathered = action
                y = np.repeat(labels[i], action['root'].shape[0])
                first = 0
            else:
                for joint_name in gathered.keys():
                    gathered[joint_name] = np.vstack((gathered[joint_name], action[joint_name]))
                y = np.concatenate((y, np.repeat(labels[i], action['root'].shape[0])))
    if big_matrix:
        first = 1
        for joint_matrix in gathered.values():
            if first:
                X = joint_matrix
                first = 0
            else:
                X = np.hstack((X, joint_matrix))
        return X, y
    return gathered, y


def get_motion_samples(selected, length, num_samples=1, random=True, sample_rate=1):
    '''Input the length of the sample in terms of frames.
    Input the selected files to be used for sampling.
    Returns 'num_samples' amount of coordinate sets of 'length' for each tracked joint in 'joints' '''
    if length is None:
        limit = None
    else:
        limit = length * num_samples * 20
    data = parse_selected(selected, limit=limit, sample_rate=sample_rate, sep_trans_root=False, motions_as_mat=False)
    actionids = [actionid for sublist in [data[key]['actionid'] for key in data.keys()] for actionid in sublist]
    actions = [action for sublist in [data[key]['actions'] for key in data.keys()] for action in sublist]
    if random == True:
        num_actions = len(actions)
        choices = np.random.choice(np.arange(num_actions), num_samples, replace=True)
        samples = []
        sampleids = []
        for choice in choices:
            action = actions[choice]
            if length < len(action):
                high_idx = np.random.randint(low=length, high=len(action))
                low_idx = high_idx - length
            else:
                high_idx = len(action)
                low_idx = 0
            sample = action[low_idx:high_idx]
            samples.append(sample)
            sampleids.append(actionids[choice])
    else:
        samples = actions
        sampleids = actionids
    return samples, sampleids


def get_lengths_np(data):
    len_array = np.array([])
    for key in data.keys():
        len_array = np.concatenate((len_array, np.array(data[key]['lengths'])))
    return len_array.astype(int)


def get_fold(data, K, type='Kfold', group='motion'):
    """Maybe change this code later to account for representation of each label in the folds"""
    len_array = get_lengths_np(data)
    N = sum(len_array)
    fold_size = N // K
    if group == 'motion':
        actionids = []
        for key in data.keys():
            actionids += data[key]['actionid']
        actionids = np.array(actionids)
        if type == 'leaveoneout':
            random.shuffle(actionids)
            return actionids
        elif type == 'Kfold':
            kfold_size = [0] * K
            folds = [[] for _ in range(K)]
            num_actions = len(actionids)
            while num_actions:
                for i in range(K):
                    if kfold_size[i] < fold_size and num_actions:
                        idx = np.random.randint(num_actions)
                        folds[i].append(actionids[idx])
                        kfold_size[i] += len_array[idx]
                        len_array, actionids = np.delete(len_array, idx), np.delete(actionids, idx)
                        num_actions -= 1
            random.shuffle(folds)
            return folds


def get_Kfold_mat(data, folds):
    """Requires data as actions as np matrix. See parse_selected"""
    len_arrays = [np.array([]) for _ in range(len(folds))]
    data_Kfold = {}
    for i, fold in enumerate(folds):
        data_Kfold[i] = {}
        data_Kfold[i]['actions'], data_Kfold[i]['labels'] = [], []
        for id in fold:
            key = id[:-3]
            idx = int(np.argwhere(np.array(data[key]['actionid']) == id))
            data_Kfold[i]['actions'].append(data[key]['actions'][idx])
            data_Kfold[i]['labels'].append(data[key]['labels'][idx])
            len_arrays[i] = np.append(len_arrays[i], data[key]['lengths'][idx])
    X, y = gather_all_np(data_Kfold)
    return X, y, len_arrays


def get_train_test_split(X, k, len_arrays, y=None):
    len_arrays_ = len_arrays.copy()
    low = int(sum(np.concatenate(len_arrays_[:k] + [np.empty(0)])))
    high = int(low + sum(len_arrays_[k]))
    len_array_test = len_arrays_.pop(k).astype(int)
    len_array_train = np.concatenate(len_arrays_).astype(int)
    Xtest = X[low:high,:]
    Xtrain = np.delete(X, slice(low, high), axis=0)
    if y is not None:
        ytest = y[low:high]
        ytrain = np.delete(y, slice(low,high))
        return Xtrain, Xtest, len_array_train, len_array_test, ytrain, ytest
    return Xtrain, Xtest, len_array_train, len_array_test


if __name__ == "__main__":
    selected = get_manual_names(["walk","run"])
