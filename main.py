
from stereo import StereoPair
import cv2

if __name__ == "__main__":
    im1 = cv2.imread("data/Backpack-perfect/im1.png")
    im2 = cv2.imread("data/Backpack-perfect/im0.png")

    stereo = StereoPair(im1, im2, "data/Backpack-perfect/calib.txt",
    plot_lines=True, resize=(480,480))
    stereo.compute_stereogram()
    stereo.normalize()
    plt.imshow(stereo.depth_im)
