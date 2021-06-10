import numpy as np
from amc_parser import *


def amc_converter(frames, filename):
    """
    :param frames: list of dictionaries specifying joint angles for each frame
    :param filename: input desired filename as string
    :return: frames as .amc file
    """
    f = open(f"Generated_motion_sequences/{filename}.amc", "w+")
    f.write(":FULLY-SPECIFIED\n")
    f.write(":DEGREES\n")
    for i in range(len(frames)):
        f.write(f"{i + 1}\n")
        for key in frames[i]:
            string = " ".join([str(f) for f in frames[i][key]])
            f.write(f"{key} {string}\n")
    f.close()
