import os
from shutil import copyfile
from pyTorch_3Dviewer import *
from pyTorch_parser import *

rootdir = './data'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        # print(os.path.join(subdir, file))
        if file[-4:] == '.amc':
            subject = file[0:3]
            amc_path = os.path.join(subdir, file)#f'./data/{subject}/{subject}_{amc_num}.amc'
            asf_path = f'./data/{subject}/{subject}.asf'

            print(file)
            joints = parse_asf(asf_path)
            motions = parse_amc(amc_path)
            v = Viewer(joints, motions)
            v.run()

            print(" 1: Walk data \n 2: Run data \n 3: Dance data \n 4: Other \n 5: Discard (buggy)")
            choice = input("Choice: ")
            if choice == '1':
                copyfile(amc_path, os.path.join("./data/Walk data", file))
            if choice == '2':
                copyfile(amc_path, os.path.join("./data/Run data",file))
            if choice == '3':
                copyfile(amc_path, os.path.join("./data/Dance data data",file))
            if choice == '4':
                copyfile(amc_path, os.path.join("./data/Other data", file))

