
from stereo import StereoBlockMatch, VectorizedStereoBlockMatch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy

if __name__ == "__main__":
    im1 = cv2.imread("data/Backpack-perfect/im1.png")
    im2 = cv2.imread("data/Backpack-perfect/im0.png")

    im1 = np.random.randint(0, 255, 16**2 * 3).reshape(16,16, 3)
    im2 = np.hstack((im1[:,4:,:], np.random.randint(0, 255, 4*16*3).reshape(16,4,3)))

    stereo1 = StereoBlockMatch(im1, im2, "data/Backpack-perfect/calib.txt", plot_lines=True, window_size=3, resize=(480,480))
    stereo1.params['ndisp'] = 11
    stereo2 = VectorizedStereoBlockMatch(im1, im2, "data/Backpack-perfect/calib.txt", window_size=3, resize=(480,480))
    stereo2.params['ndisp'] = 11
    
    stereo1.compute_stereogram()
    stereo2.compute_stereogram()
    
    #stereo1.normalize(0.05)
    #stereo2.normalize(0.05)
    
    plt.imshow(stereo1.depth_im)
    plt.imshow(stereo2.depth_im)