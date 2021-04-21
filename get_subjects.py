import os
import numpy as np
from pyTorch_parser import parse_asf, parse_amc, read_line
from amc_parser import parse_asf_np, parse_amc_np
from tqdm import tqdm

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
                # if " and " in " and ".join(matching_queries):
                #     print("BINGO", filename)
                match.append((filename, " and ".join(matching_queries)))
        if match:
            selected[subject_number] = match
    return selected


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


def get_motion_samples(selected, length, num_samples, random=True, sample_rate=1):
    '''Input the length of the sample in terms of frames.
    Input the selected files to be used for sampling.
    Returns 'num_samples' amount of coordinate sets of 'length' for each tracked joint in 'joints' '''
    data = parse_selected(selected, limit=length * num_samples * 20, sample_rate=sample_rate, sep_trans_root=False, motions_as_mat=False)
    actions = [action for sublist in [data[key]['actions'] for key in data.keys()] for action in sublist]
    num_actions = len(actions)
    choices = np.random.choice(np.arange(num_actions), num_samples, replace=True)
    samples = []
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
    return samples


def get_lengths_np(data):
    len_array = np.array([])
    for key in data.keys():
        len_array = np.concatenate((len_array, np.array(data[key]['lengths'])))
    return len_array.astype(int)

if __name__ == "__main__":
    selected = get_fnames( ["walk"] )
    # selected = get_fnames(["walk","dance"])
    samples = get_motion_samples(selected, 50, 10, sample_rate=4)
    data = parse_selected(selected, sample_rate=4, limit=3000, motions_as_mat=False)
    X, y = gather_all_np(data)
    #Save as numpy arrays for later use
    #np.save('X_walk-dance_np', X)
    #np.save('y_walk-dance_np', y) # save the file as "---.npy"