
# Python program to take
# screenshots
from operator import countOf
import numpy as np
import cv2 as cv2
from numpy.core.defchararray import count
import pyautogui
import matplotlib as plt

font = cv2.FONT_HERSHEY_COMPLEX


def takeScreenShot():
    # take screenshot using pyautogui
    image = pyautogui.screenshot()
    return cv2.cvtColor(np.array(image),
                        cv2.COLOR_RGB2BGR)


img = cv2.imread("./fourth_phase.png")  # third_phase.png
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

origin_img = img.copy()

img = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)

# ksize
ksize = (2, 2)

# Using cv2.blur() method
img_blur = cv2.blur(img, ksize)

img_gray = cv2.cvtColor(img_blur, cv2.COLOR_RGB2GRAY)  # Apply the thresholding

thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)


mask = thresh  # edges
# cv2.imshow('mask', mask)

contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
contours = sorted(contours, key=cv2.contourArea, reverse=True)
print(len(contours))

# 4, boss
# 6: barre! enfin
i = 0
for contour in contours:

    if i == 6:
        rect = cv2.minAreaRect(contour)
        print(rect)
        box = cv2.boxPoints(rect)
        print(box)
        box = box.astype('int')

        print(box)

        cv2.drawContours(origin_img, [box], -1, (0, 255, 0), 3)
        break
    i += 1


# writing it to the disk using opencv
# cv2.imwrite("./image1.png", takeScreenShot())

cv2.imshow('image', img)
a = 3
cv2.imshow('origin', origin_img)
#cv2.imshow('threshold', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
