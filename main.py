from operator import countOf
import numpy as np
import cv2 as cv2
from cv2 import imshow
from scipy import ndimage
import time
import pyautogui
from numpy.core.defchararray import count
import matplotlib.pyplot as plt


# Hyperparameters ---------------------------------------------------------------------------------

SCREENSHOT_ANGLE = -3.7

PROGRESSBAR_RATIO = 7.185566648008215
EPSILON_RATIO = 1.0

CUPHEAD_AREA = 2129.8714588528674
EPSILON_CUPHEAD_AREA = 500

CUPHEAD_RATIO = 1.0
EPSILON_CUPHEAD_RATIO = 0.5

# Helper functions --------------------------------------------------------------------------------


def takeScreenShot():
    # take screenshot using pyautogui
    image = pyautogui.screenshot()
    return cv2.cvtColor(np.array(image),
                        cv2.COLOR_RGB2BGR)


def adaptImage(img, display=False):
    # Denoising the picture
    img_denoised = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)

    # Blur the picture
    ksize = (2, 2)
    img_blur = cv2.blur(img_denoised, ksize)

    # turning the image into gray
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)

    # Apply the thresholding
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    if display:
        plt.imshow(img_denoised)
        plt.imshow(img_blur)
        plt.imshow(img_gray, cmap='gray')
        plt.imshow(thresh, cmap='gray')
        plt.show()

    return thresh


def findProgressBar(img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    new_contours = []

    for cnt in contours:
        # I have used min Area rect for better result
        rect = cv2.minAreaRect(cnt)
        width = rect[1][0]
        height = rect[1][1]
        if height == 0 or width == 0:
            continue

        ratio = width/height

        if abs(ratio - PROGRESSBAR_RATIO) < EPSILON_RATIO:
            new_contours.append(cnt)
            break

    assert len(new_contours) == 1, "more thant one contour found: found %r" % len(
        contours)

    return new_contours[0]


def get_box_from_contour(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype('int')

    return rect, box


if __name__ == "__main__":
    # Fetching original image
    display = False

    origin_img = cv2.imread("phase4.png")
    height, width, channels = origin_img.shape

    # rotation of the image
    origin_img = ndimage.rotate(origin_img, SCREENSHOT_ANGLE, reshape=False)
    if display:
        plt.imshow(origin_img)
        plt.show()

    img = origin_img.copy()

    # now we apply thresholding, bluring, etc, to the image to create a mask
    mask = adaptImage(img)

    if display:
        plt.imshow(mask)
        plt.show()

    progressbar_contour = findProgressBar(mask)

    progressbar_rect, progressbar_box = get_box_from_contour(
        progressbar_contour)
