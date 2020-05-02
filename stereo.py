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
        self.window_size = window_size
        self.half_window_size = window_size // 2
        self.r, self.c, _ = self.im1.shape
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
        """
        wrapper around _compute_stereogram, in case you want to compute the stereogram both ways
        i.e. im1 -> im2 and im2 -> im1
        """
        self._compute_stereogram(self.im1, self.im2)


    def _compute_stereogram(self, im1, im2):
        """
        subclasses implement different stereo algorithms
        """
        assert False, "implement in subclass"


    
    def compute_depth(self, offset):
        """
        given offset of a point computes depth
        """
        return (self.params['focal_length'] * self.params['baseline']) / (offset + 0.01)


    def normalize(self, q):
        """
        replace all values less than quantile q with quantile q
        replace all values greater than quantile 1-1 with quantile 1-q

        """
        lower, upper = tuple(np.quantile(self.depth_im, (q, 1-q)))
        self.depth_im[self.depth_im < lower] = lower
        self.depth_im[self.depth_im > upper] = upper

    
    def mse(self, cutout_a, cutout_b):
        """
        compute mse between two cutouts
        """
        diff = np.float32(cutout_a) - np.float32(cutout_b)
        diff **= 2
        return np.mean(diff)
    
    def save_stereogram(self, imname):
        cv2.imwrite(imname, self.depth_im)






class StereoBlockMatch(_BasicStereo):

    def _compute_stereogram(self, im1, im2):
        """
        computes stereogram from two images using naive (unvectorized)
        implementation of block matching algorithm
        slow af
        works ok
        """
        max_displacement = int(self.params['ndisp'])
        shift_im = np.zeros(self.r*self.c).reshape((self.r, self.c))
        
        # (i,j) loop over each pixel in image1 
        for i in range(self.half_window_size, self.r - self.half_window_size, self.stride_y):
            print(i)
            for j in range(self.half_window_size, self.c - self.half_window_size, self.stride_x):
                
                # take cutout centered on each pixel in image1
                cutout = im1[i-self.half_window_size:i+self.half_window_size+1,
                j-self.half_window_size:j+self.half_window_size+1,:]
                
                # search along epipolar line for matching cutout
                # limit search to max_displacement pixels in each direction
                mse_errors = []
                indices = []
                for k in range(max(self.half_window_size, j - max_displacement), min(im1.shape[1] - self.half_window_size, j + max_displacement)):
                    cutout_match = im2[i-self.half_window_size:i+self.half_window_size+1,
                    k-self.half_window_size:k+self.half_window_size+1,:]
                    mse_errors.append(self.mse(cutout, cutout_match))
                    indices.append(k)
                
                min_ind = np.argmin(mse_errors)
                min_ind = indices[min_ind]
                shift_im[i,j] = j-min_ind
                depth_est = self.compute_depth(j - min_ind)
                self.depth_im[i,j] = depth_est
                

                # save curve of mse matches for this row
                if self.plot_lines:
                    if i % 30 == 0 and j in self.j_indices:
                        self.lines.append(cutout_match)
        #self.depth_im = self.compute_depth(shift_im)
        self.depth_im = self.depth_im[1:-1,1:-1]
        np.save("basicStereo.npy", shift_im)
        #pdb.set_trace()



class VectorizedStereoBlockMatch(_BasicStereo):

    def _compute_stereogram(self, im1, im2):
        
        max_displacement = int(self.params['ndisp'])
        mse_list = []
        
        # shift image by max displacement in both directions
        for i in reversed(range(-max_displacement, max_displacement)):
            print(i)
            if i != 0:
                if i < 0:
                    shifted_im2 = im2[:, :i].copy() # cut off right
                    shifted_im1 = im1[:, -i:].copy() # cut off left
                else:
                    shifted_im2 = im2[:, i:].copy() # cut off left
                    shifted_im1 = im1[:, :-i].copy() # cut off right
            else:
                shifted_im1 = im1.copy()
                shifted_im2 = im2.copy()

            mse_im = cv2.subtract(shifted_im1, shifted_im2) ** 2
            mse_im = np.mean(mse_im, axis=2)
            mse_integral = cv2.integral(mse_im)
            #mse_integral = mse_integral[:-1,1:]
            
            # use formula to compute sum of cutout from image integral
            # bottomright + topright - (bottomleft + topright)
            # each pixel of mse array contains mse in windowsize x windowsize
            # window around it
            w = self.window_size
            A = mse_integral[w:, w:] + \
                mse_integral[:-w, :-w]
            B = mse_integral[w:, :-w] + \
                mse_integral[:-w, w:]
            mse = (A - B) / (self.window_size ** 2)
            mse = np.float32(mse)
            shift_amount = np.abs(i)
            
            # pad so that mse arrays can be concatenated into one 3d array
            # and np.argmin can be used
            if i < 0:
                mse = self.pad_with_inf(mse, "left", shift_amount)
            else:
                mse = self.pad_with_inf(mse, "right", shift_amount)
            mse_list.append(mse)
        mse_list = np.stack(mse_list)
        mse_list = np.argmin(mse_list, axis=0)
        
        # get distance of each offset from offset=0
        mse_list -= max_displacement
        padding = np.zeros(im1.shape[1]).reshape(-1,im1.shape[1])
        #mse_list = np.vstack((padding, mse_list))
        #mse_list = np.vstack((mse_list, padding))
        #mse_list[:,0] = padding
        #mse_list[:,-1] = padding
        #mse_list = np.float64(mse_list)
        mse_list = np.int32(mse_list)
        np.save("vectorizedStereo.npy", mse_list)
        self.depth_im = self.compute_depth(mse_list)

        



    
    def pad_with_inf(self, img, direction, padding):
        """
        pad im to left or right with array of shape (im.shape[0], padding) of inf's
        """
        assert direction in {'left', 'right'}, "must pad to the left or right of image"
        pad = np.array([float('inf')] * (img.shape[0]*padding)).reshape(img.shape[0], padding)
        if direction == "left":
            img = np.hstack((pad,img))
        elif direction == "right":
            img = np.hstack((img, pad))
        return img





            


        


