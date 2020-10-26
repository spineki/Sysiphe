import time
import keyboard
import pyautogui

import cv2 as cv2
from scipy import ndimage
import numpy as np
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


def display_contour_on_img(box, img):
    display_img = img.copy()
    cv2.drawContours(display_img, [box], -1, (0, 255, 0), 3)
    plt.imshow(display_img)
    plt.show()


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

    if len(new_contours) != 1:
        print("more thant one contour for progressbar found: found %r" % len(
            contours))
        return None
    return new_contours[0]


def findCuphead(img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    new_contours = []

    i = -1
    for cnt in contours:
        i += 1
        # I have used min Area rect for better result
        rect = cv2.minAreaRect(cnt)
        width = rect[1][0]
        height = rect[1][1]
        area = width * height
        if height == 0 or width == 0:
            continue
        ratio = width/height
        if abs(ratio - CUPHEAD_RATIO) < EPSILON_CUPHEAD_RATIO and abs(area - CUPHEAD_AREA) < EPSILON_CUPHEAD_AREA:
            _, box = get_box_from_contour(contours[i])
            new_contours.append(cnt)
            break

    if len(new_contours) != 1:
        print("more thant one contour for cuphead found: found %r" % len(
            contours))
        return None

    return new_contours[0]


def get_box_from_contour(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype('int')

    return rect, box


def progression_pipeline(img, display=False):
    """
    take a picture as input, return progression as output

    @param: img a opencv image
    @param: display: boolean: whether to display intermediates pictures or not

    @return: int, progression, or None if an error occured
    """
    # now we apply thresholding, bluring, etc, to the image to create a mask
    mask = adaptImage(img)

    if display:
        plt.imshow(mask)
        plt.show()

    # we get the contour of the progressbar
    progressbar_contour = findProgressBar(mask)
    if progressbar_contour is None:
        return None

    # we extract the bounding box from it
    progressbar_rect, progressbar_box = get_box_from_contour(
        progressbar_contour)

    if display:
        display_contour_on_img(progressbar_box, origin_img)

    # no need to work with the full picture, we can safely crop the progressbar
    x, y, w, h = cv2.boundingRect(progressbar_contour)
    cropped_progressbar = mask[y:y+h, x:x+w]
    if display:
        plt.imshow(cropped_progressbar, cmap='gray')
        plt.show()

    # we get the contour of cuphead
    cuphead_contour = findCuphead(cropped_progressbar)
    if cuphead_contour is None:
        return None

    # We extract the bounding box from it
    cuphead_rect, cuphead_box = get_box_from_contour(cuphead_contour)
    if display:
        display_contour_on_img(cuphead_box, cropped_progressbar)

    progressbar_w = progressbar_rect[1][0]
    cuphead_x = cuphead_rect[0][0]
    cuphead_w = cuphead_rect[1][0]

    relative_progress = (cuphead_x + cuphead_w/2) / progressbar_w

    return int(relative_progress * 100)


if __name__ == "__main__":
    # Fetching original image

    mode = "TEST"
    mode = "GAME"  # comment this line to go to test mode
    display = True

    if mode == "TEST":
        origin_img = cv2.imread("phase3.png")
        height, width, channels = origin_img.shape

        # rotation of the image
        origin_img = ndimage.rotate(
            origin_img, SCREENSHOT_ANGLE, reshape=False)
        if display:
            plt.imshow(origin_img)
            plt.show()

        img = origin_img.copy()

        my_timer = time.perf_counter()

        progression = progression_pipeline(img, display=display)

        print("done in {} s".format(time.perf_counter() - my_timer))

        print(progression, "%")

    else:
        while True:  # making a loop
            time.sleep(0.16)  # pause to avoid burning the cpu
            try:  # used try so that if user pressed other than the given key error will not be shown
                if keyboard.is_pressed('space'):  # if key 'q' is pressed
                    print('Space pressed, launching the program')
                    origin_img = takeScreenShot()
                    origin_img = ndimage.rotate(
                        origin_img, SCREENSHOT_ANGLE, reshape=False)

                    img = origin_img.copy()

                    my_timer = time.perf_counter()

                    progression = progression_pipeline(img, display=display)

                    print("done in {} s".format(time.perf_counter() - my_timer))

                    print(progression, "%")

            except Exception as e:
                pass
