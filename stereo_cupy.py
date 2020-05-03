import cupy as cp
import numpy as np
import cv2
from stereo import _BasicStereo
import pdb

class CudaStereoBlockMatch(_BasicStereo):

    def _compute_stereogram(self, im1, im2):
        
        max_displacement = int(self.params['ndisp'])
        mse_list = []
        
        # shift image by max displacement in both directions
        for i in reversed(range(-max_displacement, max_displacement)):
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

            mse_im = (cp.float32(shifted_im1) - cp.float32(shifted_im2)) ** 2
            mse_im = cp.mean(mse_im, axis=2)
            mse_integral = cv2.integral(mse_im)
            
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
            mse = cp.float32(mse)
            shift_amount = np.abs(i)
            
            # pad so that mse arrays can be concatenated into one 3d array
            # and cp.argmin can be used
            if i < 0:
                mse = self.pad_with_inf(mse, "left", shift_amount)
            else:
                mse = self.pad_with_inf(mse, "right", shift_amount)
            mse_list.append(mse)
        mse_list = cp.stack(mse_list)
        cp.save("vectorizedStereo.npy", mse_list)
        mse_list = cp.argmin(mse_list, axis=0)
        
        # get distance of each offset from offset=0
        mse_list -= max_displacement
        padding = cp.zeros(im1.shape[1]).reshape(-1,im1.shape[1])
        mse_list = mse_list.astype(cp.int32)
        self.depth_im = self.compute_depth(mse_list)
        self.depth_im = cp.asnumpy(self.depth_im)

        
    def pad_with_inf(self, img, direction, padding):
        """
        pad im to left or right with array of shape (im.shape[0], padding) of inf's
        """
        assert direction in {'left', 'right'}, "must pad to the left or right of image"
        pad = cp.array([float('inf')] * (img.shape[0]*padding)).reshape(img.shape[0], padding)
        if direction == "left":
            img = cp.hstack((pad,img))
        elif direction == "right":
            img = cp.hstack((img, pad))
        return img