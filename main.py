
from BlockMatching import StereoBlockMatch, VectorizedStereoBlockMatch
from SemiGlobalMatching import SemiGlobalMatching
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import time
import pdb
try:
    from BlockMatchingCupy import CudaStereoBlockMatch
except ImportError:
    pass



IMAGE_DIR = "Bicycle1-perfect"

if __name__ == "__main__":
    im1 = cv2.imread(os.path.join("data", IMAGE_DIR ,"im1.png"))
    im2 = cv2.imread(os.path.join("data", IMAGE_DIR ,"im0.png"))
    #im1 = im1.astype(np.int32)
    #im2 = im2.astype(np.int32)

    #stereo1 = StereoBlockMatch(im1, im2, "data/Adirondack-perfect/calib.txt", plot_lines=True, window_size=11, resize=(480,480))
    #stereo2 = VectorizedStereoBlockMatch(im1, im2, "data/Adirondack-perfect/calib.txt", window_size=11, resize=(480,480))
    
    # uncomment if you have a gpu
    #stereo3 = CudaStereoBlockMatch(im1, im2, "data/Backpack-perfect/calib.txt", window_size=11, resize=(480,480))
    stereo4 = SemiGlobalMatching(im1, im2, os.path.join("data", IMAGE_DIR ,"calib.txt"),
    window_size=3, resize=(480,480))

    # stereo1.params['ndisp'] = 50
    # stereo2.params['ndisp'] = 50
    # stereo4.params['ndisp'] = 50

    # t1 = time.time()
    # stereo1.compute_stereogram()
    # print("naive block matching time {:.2f}".format(time.time() - t1))
    # stereo1.normalize(0.05)
    # plt.imshow(stereo1.depth_im)
    
    # t1 = time.time()
    # stereo2.compute_stereogram()
    # print("vectorized block matching time {:.2f}".format(time.time() - t1))
    # stereo2.normalize(0.05)
    # plt.imshow(stereo2.depth_im)
    #plt.savefig("vectorized.png")
    #stereo2.save_stereogram("vectorized.png")
    #stereo3.compute_stereogram()

    params = {"p1":5, "p2":30000, "census_kernel_size":7, "reversed":True}
    stereo4.set_params(params)
    stereo4.params['ndisp'] = 50
    t1 = time.time()
    im = stereo4.compute_stereogram()
    out = stereo4.normalize(im, 0.1)
    out = cv2.medianBlur(out.astype(np.float32), 5)
    print("sgbm time {:.2f}".format(time.time() - t1))
    plt.imshow(out)