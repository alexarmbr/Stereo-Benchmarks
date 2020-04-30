import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb


class _BasicStereo:

    def __init__(self, im1, im2, metadata_path,
    window_size = 11, stride_x = 1, stride_y = 1,
    plot_lines = False,resize = None):
        """
        im1, im2: two images taken by calibrated camera
        
        focal_length: focal length of camera used to take images,
        
        B: baseline, distance between camera when two images were taken. It is
        assumed that there is no vertical shift, i.e. camera was moved only in x direction
        
        metadata_path: path to file containing camera matrix, and ndisp value
        
        window_size: size of matching window, large window = more computation, should be odd
        
        stride_y,x: how many pixels to skip in between matching computation
        """

        self.im1 = im1
        self.im2 = im2
        if resize is not None:
            self.im1 = cv2.resize(self.im1, resize)
            self.im2 = cv2.resize(self.im2, resize)

        self.stride_x = stride_x
        self.stride_y = stride_y
        assert self.im1.shape == self.im2.shape, "image shapes must match exactly"
        assert window_size % 2 == 1, "window size should be odd number"
        self.window_size = window_size // 2
        self.r, self.c,_ = self.im1.shape
        self.params = self.parse_metadata(metadata_path)
        self.depth_im = np.zeros(self.r*self.c).reshape((self.r, self.c))
        self.plot_lines = plot_lines

        if self.plot_lines:
            self.j_indices = np.random.random_integers(0, self.c, 20)
            self.lines = []


    def parse_metadata(self, filename):
        d = {}
        with open(filename) as f:
            for line in f:
                (key, val) = line.split("=")
                val = val.strip("\n")
                try:
                    val = float(val)
                except:
                    pass
                d[key]=val
        d['focal_length'] = float(d['cam0'].strip("[").split(" ")[0])
        return d

    
    def compute_stereogram(self):
        self._compute_stereogram(self.im1, self.im2)


    def _compute_stereogram(self, im1, im2):
        """
        subclasses implement different stereo algorithms
        """
        raise NotImplementedError


    
    def compute_depth(self, offset):
        """
        given offset of a point computes depth
        """
        return (self.params['focal_length'] * self.params['baseline']) / (offset + 0.01)


    def normalize(self):
        lower, upper = tuple(np.quantile(self.depth_im, (0.05, 0.95)))
        self.depth_im[depth_im < lower] = lower
        self.depth_im[depth_im > upper] = upper

    
    def mse(self, cutout_a, cutout_b):
        """
        compute mse between two cutouts
        """
        diff = np.float32(cutout_a) - np.float32(cutout_b)
        diff **=2
        return np.mean(diff)








class StereoBlockMatch(_BasicStereo)


    def _compute_stereogram(self, im1, im2):
        """
        computes stereogram from two images using block matching algorithm
        slow af
        """
        max_displacement = int(self.params['ndisp'])
        for i in range(self.window_size, self.r - self.window_size, self.stride_y):
            print(i)
            for j in range(self.window_size, self.c - self.window_size, self.stride_x):
                cutout = im1[i-self.window_size:i+self.window_size,
                j-self.window_size:j+self.window_size,:]
                
                # search along epipolar line for matching cutout
                # limit search to max_displacement pixels in each direction
                mse_errors = []
                indices = []
                for k in range(max(self.window_size, j - max_displacement), min(im1.shape[1] - self.window_size, j + max_displacement)):
                    cutout_match = im2[i-self.window_size:i+self.window_size,
                    k-self.window_size:k+self.window_size,:]
                    mse_errors.append(self.mse(cutout, cutout_match))
                    indices.append(k)
                
                min_ind = np.argmin(mse_errors)
                min_ind = indices[min_ind]
                depth_est = self.compute_depth(j - min_ind)
                self.depth_im[i,j] = depth_est

                # save curve of mse matches for this row
                if self.plot_lines:
                    if i % 30 == 0 and j in self.j_indices:
                        self.lines.append(cutout_match)