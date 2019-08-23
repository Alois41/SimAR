import cv2
import imutils
import numpy as np
from source.configuration import Config as p


def adjust_gamma(image, gamma=2.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def crop_zone(image, x0, y0, xf, yf):
    return image[int(y0):int(yf), int(x0):int(xf)]


def zoom_center(image):
    """ zoom in the center of the image"""
    crop_img = image[p.cam_area[0][0]:p.cam_area[1][0], p.cam_area[0][1]:p.cam_area[1][1]]
    return imutils.resize(crop_img, width=image.shape[1])


def find_center_contours(image):
    """ find all the contours in the center of a given image"""

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh_rgb = blurred.copy()

    # for each color channel
    for index, channel in enumerate(cv2.split(thresh_rgb)):
        # make channel binary with an adaptive threshold
        thresh = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 103, 8)
        # Structure the channel with Rectangle shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # merge channels
        thresh_rgb[:, :, index] = thresh

    # convert into Gray scale(luminance)
    thresh_gray = cv2.cvtColor(thresh_rgb, cv2.COLOR_RGB2GRAY)
    # zoom in the center
    thresh_gray = zoom_center(thresh_gray)
    image = zoom_center(image)
    # invert black/white
    thresh_gray = cv2.bitwise_not(thresh_gray)

    kernel = np.ones((5, 5), np.uint8)
    thresh_gray = cv2.erode(thresh_gray, kernel, iterations=10)

    # find contours
    contours = cv2.findContours(thresh_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    return contours, image
