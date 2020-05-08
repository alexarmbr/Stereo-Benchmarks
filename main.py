
from stereo import StereoBlockMatch, VectorizedStereoBlockMatch
from SemiGlobalMatching import SemiGlobalMatching
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import pdb
try:
    from stereo_cupy import CudaStereoBlockMatch
except ImportError:
    pass





if __name__ == "__main__":
    im1 = cv2.imread("data/Bicycle1-perfect/im1.png")
    im2 = cv2.imread("data/Bicycle1-perfect/im0.png")
    
    
    stereo4 = SemiGlobalMatching(im1, im2, "data/Backpack-perfect/calib.txt",
    window_size=3, resize=(240,240))
    params = {"p1":5, "p2":300, "census_kernel_size":7, "reversed":True}
    stereo4.set_params(params)
    stereo4.params['ndisp'] = 50
    stereogram = stereo4.compute_stereogram()