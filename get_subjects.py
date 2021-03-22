import os
import numpy as np
from pyTorch_parser import parse_asf, parse_amc, read_line
from amc_parser import parse_asf_np, parse_amc_np


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


def parse_selected(selected, type="numpy", relative_sample_rate=1, limit=None, ignore_translation=True, motions_as_mat=True):
    count = 0
    data = {}
    for key, file in selected.items():
        asf_path = f'./data/{key}/{key}.asf'
        if type == "numpy":
            joints = parse_asf_np(asf_path)
        else:
            joints = parse_asf(asf_path)

        actions = []
        labels = []
        for filename, label in file:
            amc_path = f'./data/{key}/{filename}.amc'
            if type == "numpy":
                motions = parse_amc_np(amc_path, ignore_translation)
            else:
                motions = parse_amc(amc_path, ignore_translation)

            sampled_motions = []
            for i, motion in enumerate(motions):
                if not i % relative_sample_rate:
                    motions[i]['translation'] = motion['root'][:3]
                    motions[i]['root'] = motion['root'][3:]
                    sampled_motions.append(motions[i])
            motions = sampled_motions
            if limit:
                count += len(motions)
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
        data[key] = {}
        data[key]['joints'] = joints
        data[key]['actions'] = actions
        data[key]['labels'] = labels
        if limit:
            if count >= limit:
                return data
    return data


def gather_all_np(data ,big_matrix=True):
    """Requires data as actions as np matrix. See parse_selected"""
    first = 1
    for set in data.values():
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


if __name__ == "__main__":
    selected = get_fnames( ["walk"] )
    # selected = get_fnames(["walk","dance"])
    data = parse_selected(selected, relative_sample_rate=4, limit=3000)
    X, y = gather_all_np(data)
    123
    #Save as numpy arrays for later use
    #np.save('X_walk-dance_np', X)
    #np.save('y_walk-dance_np', y) # save the file as "---.npy"
