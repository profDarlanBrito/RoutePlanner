from numpy import ndarray
import numpy as np


def ConvertArray2String(fileCA2S, array: ndarray):
    np.set_printoptions(threshold=10000000000)
    np.savetxt(fileCA2S, array, fmt="%.3f", delimiter=" ")
    return fileCA2S
