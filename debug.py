
np.random.seed(1)
im1 = np.random.randint(0, 255, 16**2 * 3).reshape(16,16, 3)
im2 = np.hstack((im1[:,4:,:], np.random.randint(0, 255, 4*16*3).reshape(16,4,3)))



max_displacement = 11
r,c,_ = im1.shape
shift_im = np.zeros(r*c).reshape((r, c))

# (i,j) loop over each pixel in image1 
for i in range(half_window_size, r - half_window_size, stride_y):
    print(i)
    for j in range(half_window_size, c - half_window_size, stride_x):
        
        # take cutout centered on each pixel in image1
        cutout = im1[i-half_window_size:i+half_window_size,
        j-half_window_size:j+half_window_size,:]
        
        # search along epipolar line for matching cutout
        # limit search to max_displacement pixels in each direction
        mse_errors = []
        indices = []
        for k in range(max(half_window_size, j - max_displacement), min(im1.shape[1] - half_window_size, j + max_displacement)):
            cutout_match = im2[i-half_window_size:i+half_window_size,
            k-half_window_size:k+half_window_size,:]
            mse_errors.append(mse(cutout, cutout_match))
            indices.append(k)
        
        min_ind = np.argmin(mse_errors)
        min_ind = indices[min_ind]
        shift_im[i,j] = j-min_ind
        depth_est = compute_depth(j - min_ind)
        depth_im[i,j] = depth_est

        # save curve of mse matches for this row
        if plot_lines:
            if i % 30 == 0 and j in j_indices:
                lines.append(cutout_match)





np.random.seed(1)
im1 = np.random.randint(0, 255, 16**2 * 3).reshape(16,16, 3)
im2 = np.hstack((im1[:,4:,:], np.random.randint(0, 255, 4*16*3).reshape(16,4,3)))


max_displacement = 11
mse_list = []
window_size = 3


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
    mse_integral = mse_integral[:-1,1:]
    
    # use formula to compute sum of cutout from image integral
    # bottomright + topright - (bottomleft + topright)
    # each pixel of mse array contains mse in windowsize x windowsize
    # window around it
    w = window_size - 1
    A = mse_integral[w:, w:] + \
        mse_integral[:-w, :-w]
    B = mse_integral[w:, :-w] + \
        mse_integral[:-w, w:]
    mse = (A - B) / (window_size ** 2)
    mse = np.float32(mse)
    shift_amount = np.abs(i)
    
    # pad so that mse arrays can be concatenated into one 3d array
    # and np.argmin can be used
    if i < 0:
        mse = pad_with_inf(mse, "left", shift_amount)
    else:
        mse = pad_with_inf(mse, "right", shift_amount)
    mse_list.append(mse)
mse_list = np.stack(mse_list)
mse_list = np.argmin(mse_list, axis=0)

# get distance of each offset from offset=0
mse_list -= max_displacement
padding = np.zeros(im1.shape[1]).reshape(-1,im1.shape[1])
mse_list = np.int32(mse_list)
np.save("vectorizedStereo.npy", mse_list)
depth_im = compute_depth(mse_list)






def pad_with_inf(img, direction, padding):
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






# image integral test
# make sure image integral is equivalent to sliding window kernel
K = 5
L = 10
test_arr = np.random.uniform(0, 100, 100).reshape(L,L)
arr = test_arr.copy()

lst = []
k = K // 2
for i in range(k, L-k):
    for j in range(k, L-k):
        lst.append(np.mean(arr[i-k:i+k+1, j-k:j+k+1]))
smoothed = np.array(lst).reshape(L-2*k, L-2*k)

smoothed_2 = cv2.blur(test_arr, (K,K))
np.isclose(smoothed, smoothed_2[2:-2, 2:-2])







mse_integral = cv2.integral(test_arr.copy())
#mse_integral = mse_integral[1:,1:]
# use formula to compute sum of cutout from image integral
# bottomright + topright - (bottomleft + topright)
# each pixel of mse array contains mse in windowsize x windowsize
# window around it
w = K - 1
A = mse_integral[K:, K:] + \
    mse_integral[:-K, :-K]
B = mse_integral[K:, :-K] + \
    mse_integral[:-K, K:]
mse = (A - B) / (K ** 2)
np.isclose(smoothed, mse)
