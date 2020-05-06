from stereo import _BasicStereo
import pdb
import cv2
import numpy as np

class SemiGlobalMatching(_BasicStereo):

    def __init__(self, *args, **kwargs):
        """
        Semi Global Matching stereo algorithm with hamming distance
        https://core.ac.uk/download/pdf/11134866.pdf

        Arguments:
            census_kernel_size {int} -- kernel size used to create census image
            
        """

        if "census_window_size" in kwargs:
            self.census_kernel_size = kwargs["census_window_size"]
        else:
            self.census_kernel_size = 5
        kwargs.pop("census_window_size")
        super().__init__(*args, **kwargs)

        self.im1 = cv2.cvtColor(self.im1, cv2.COLOR_BGR2GRAY)
        self.im2 = cv2.cvtColor(self.im2, cv2.COLOR_BGR2GRAY)
        assert self.census_kernel_size % 2 == 1\
             and self.census_kernel_size < 8,\
                  "census kernel size needs to odd and less than 8"
        self.csize = self.census_kernel_size // 2
        self.census_images = {}

    
    def _compute_stereogram(self, im1, im2):
        """
        compute disparity image that warps im2 -> im1

        Arguments:
            im1 {np.ndarray} -- image 1
            im2 {np.ndarray} -- image 2
        """
        cim1 = self.census_transform(im1)
        cim2 = self.census_transform(im2)

        cost_images = []
        for d in range(max_displacement):
            shifted_im2 = cim2[:, i:].copy() # cut off left
            shifted_im1 = cim1[:, :-i].copy() # cut off right
            cost_im = self.hamming_distance(shifted_im1, shifted_im2)

            self.pad_with_inf(cost_im, "right", i)
            cost_images.append(cost_im)
        
        cost_images = np.stack(cost_images)
        cost_images = self.aggregate_cost(cost_images)
        min_cost_im = np.argmin(cost_images, axis=0)
        self.depth_im = self.compute_depth(min_cost_im)




    def census_transform(self, image, imname = None):
        """
        compute census image using kernel of size csize

        Arguments:
            image {np.ndarray} -- greyscale image to compute census image of
            imname {string} -- name for image to save census image as
        """
        census_image = np.zeros(image.shape)

        for i in range(self.csize, image.shape[0] - self.csize):
            for j in range(self.csize, image.shape[1] - self.csize):
                cutout = image[i-self.csize:i+self.csize+1, j-self.csize:j+self.csize+1]
                mid = cutout[self.csize, self.csize]
                census = cutout >= mid # which surrounding pixels are greater than this pixel
                # this mask is transformed to a binary number which is used as a signature for this pixel
                 
                census = census.reshape(1,-1).squeeze()
                mid = len(census) // 2
                census = np.delete(census, mid) # remove middle element
                census_val = np.uint64(0)

                one = np.uint64(1)
                zero = np.uint64(0)
                for B in census:
                    census_val <<= one
                    census_val |= one if B else zero
                census_image[i,j] = census_val
        census_image = np.uint64(census_image)
        return census_image


    def hamming_distance(self, cim1, cim2):
        """
        Compute elementwise hamming distance between two census images,
        each pixel in the image is treated as a binary number

        Arguments:
            cim1 {np.ndarray} -- census image 1
            cim2 {np.ndarray} -- census image 2
        """
        assert cim1.shape == cim2.shape, "inputs must have same shape"
        z = np.zeros(cim1.shape)
        xor = np.bitwise_xor(cim1, cim2)
        
        while not (xor == 0).all():
            z+=xor & 1
            xor = xor >> 1
        return z



    def aggregate_cost(self, cost_array):
        """
        aggregate cost over 8 paths using dp algorithm

        Arguments:
            cost_array {np.ndarray} -- array of shape (h,w,d) that contains pixel wise costs (hamming distances) for each d
        """
        raise NotImplementedError

    








