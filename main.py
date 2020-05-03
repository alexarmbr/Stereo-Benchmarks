
from stereo import StereoBlockMatch, VectorizedStereoBlockMatch
from stereo_cupy import CudaStereoBlockMatch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time

if __name__ == "__main__":
    im1 = cv2.imread("data/Backpack-perfect/im1.png")
    im2 = cv2.imread("data/Backpack-perfect/im0.png")

    #im1 = np.random.randint(0, 255, 16**2 * 3).reshape(16,16, 3)
    #im2 = np.hstack((im1[:,4:,:], np.random.randint(0, 255, 4*16*3).reshape(16,4,3)))

    #stereo1 = StereoBlockMatch(im1, im2, "data/Backpack-perfect/calib.txt", plot_lines=True, window_size=11, resize=(480,480))
    #stereo1.params['ndisp'] = 11
    stereo2 = VectorizedStereoBlockMatch(im1, im2, "data/Backpack-perfect/calib.txt", window_size=19, resize=(480,480))
    #stereo2.params['ndisp'] = 11
    stereo3 = CudaStereoBlockMatch(im1, im2, "data/Backpack-perfect/calib.txt", window_size=11, resize=(480,480))
    
    #stereo1.compute_stereogram()
    t1 = time.time()
    stereo2.compute_stereogram()
    print("Vectorized Stereo time: %f" % (time.time() - t1))
    #stereo3.compute_stereogram()

    #stereo1.save_stereogram("stereoSlow.npy")
    #stereo1.normalize(0.05)
    #stereo2.normalize(0.07)
    t2 = time.time()
    stereo3.compute_stereogram()
    print("Cuda stereo time: %f" % (time.time() - t2))

    #plt.imshow(stereo1.depth_im)