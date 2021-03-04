import os
import numpy as np
import torch
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
            if i not in [0,1,2]:
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

def get_data( query, limit = 0, subjects = get_subjects() ):
    selected = {}
    for subject_number, file in subjects.items():
        match = []
        for filename, description in file.items():
            if limit and query == description:
                match.append(filename)
            elif query in description:
                match.append(filename)
        if len(match):
            selected[subject_number] = match
    return selected

def parse_selected( selected, type = "numpy", ignore_translation = True):
    data = {}
    for key, filenames in selected.items():
        asf_path = f'./data/{key}/{key}.asf'
        if type == "numpy":
            joints = parse_asf_np( asf_path )
        else:
            joints = parse_asf( asf_path )

        actions = []
        for filename in filenames:
            amc_path = f'./data/{key}/{filename}.amc'
            if type == "numpy":
                motions = parse_amc_np( amc_path, ignore_translation )
            else:
                motions = parse_amc( amc_path, ignore_translation )

            for i, motion in enumerate(motions):
                motions[i]['translation'] = motions[i]['root'][:3]
                motions[i]['root'] = motions[i]['root'][3:]
            actions.append(motions)
        data[key] = {}
        data[key]['joints'] = joints
        data[key]['actions'] = actions
    return data

if __name__ == "__main__":
    selected = get_data( "walk" )
    data = parse_selected( selected )
    123