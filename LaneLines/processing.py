import os
import math

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def read_image(fname):
    return mpimg.imread(fname)


def yellow2white(image):
    temp_image = np.copy(image)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(temp_image, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    lower = np.array([25, 146, 190], dtype="uint8")

    # upper = np.array([62, 174, 250], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)

    temp_image[mask != 0, :] = [255, 255, 255]

    ####
    lower = np.array([0, 0, 150], dtype="uint8")

    # upper = np.array([62, 174, 250], dtype="uint8")
    upper = np.array([255, 15, 255], dtype="uint8")

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)

    temp_image[mask != 0, :] = [255, 255, 255]
    ####
    # # Bitwise-AND mask and original image
    # res = cv2.bitwise_and(image, image, mask=mask)
    # plt.imshow(image)
    # plt.show()
    return temp_image


def grayscale(img):
    """
    Applies the Gray scale transform
    This will return an image with only one color channel
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_fitted_line(img, two_line_coeffs, ROI, color=[255, 0, 0], thickness=2):
    """
    Draw lines of any degree using given coefficients
    """
    xp = np.linspace(ROI[0][0][0], ROI[0][3][0], ROI[0][3][0] - ROI[0][0][0])

    for poly in two_line_coeffs:
        line_y = np.array(poly(xp))
        coords = np.array([xp, line_y]).T.reshape((-1, 1, 2)).astype(int)
        coords = coords[((coords[:, :, 1] > ROI[0][1][1]) & (coords[:, :, 1] < ROI[0][0][1]))[:, 0]]
        cv2.polylines(img, coords, True, color, thickness=thickness)


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Drawing of the lines given in the `lines` list
    Drawing is performed inplace on the variable `img`
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs((y2 - y1) / (x2 - x1)) > 0.15:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_side_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    imshape = img.shape
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > 0.15:
                if slope < 0:
                    left_lines.append(line)
                elif slope > 0:
                    right_lines.append(line)
    for line in [np.array(left_lines), np.array(right_lines)]:
        for x1, y1, x2, y2 in line.mean(axis=0):
            slope = (y2 - y1) / (x2 - x1)
            bias = y1 - slope * x1
            y_bottom = int(imshape[0])
            x_bottom = int((y_bottom - bias) / slope)
            y_top = int(imshape[0] * 0.6)
            x_top = int((y_top - bias) / slope)
            cv2.line(img, (x_bottom, y_bottom), (x_top, y_top), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def hough_side_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_side_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


def extrapolate_poly(imshape, lines, degree=1):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > 0.15:
                if slope < 0:
                    left_lines.append(line)
                elif slope > 0:
                    right_lines.append(line)

    left_line_points = np.array(left_lines).reshape((-1, 2)).T
    right_line_points = np.array(right_lines).reshape((-1, 2)).T

    coefs_left = np.polyfit(left_line_points[0], left_line_points[1], deg=degree)
    coefs_right = np.polyfit(right_line_points[0], right_line_points[1], deg=degree)

    # xp = np.linspace(100, 800, 700)
    # left_poly, right_poly = np.poly1d(coefs_left), np.poly1d(coefs_right)
    # left_line = left_poly(xp)
    # right_line = right_poly(xp)
    #
    # plt.plot(xp, right_line)
    # plt.plot(xp, left_line)
    # plt.plot(left_line_points[0], left_line_points[1], 'r.')
    # plt.plot(right_line_points[0], right_line_points[1], 'g.')
    # plt.show()

    return coefs_left, coefs_right


def extrapolate_poly_2(imshape, lines, degree=1):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) > 0.2:
                if slope < 0:
                    left_lines.append(line)
                elif slope > 0:
                    right_lines.append(line)

    left_line_points = np.array(left_lines).reshape((-1, 2)).T
    right_line_points = np.array(right_lines).reshape((-1, 2)).T

    left_half = left_line_points[:, :int(left_line_points.shape[1] / 2)]
    right_half = right_line_points[:, :int(right_line_points.shape[1] / 2)]

    left_half_coef = np.poly1d(np.polyfit(left_half[0], left_half[1], deg=1))
    right_half_coef = np.poly1d(np.polyfit(right_half[0], right_half[1], deg=1))
    xp_left = np.linspace(1, int(imshape[1] * 0.2), int(imshape[1] * 0.05))
    left_line = left_half_coef(xp_left)

    xp_right = np.linspace(imshape[1] - int(imshape[1] * 0.2), imshape[1], int(imshape[1] * 0.05))
    right_line = right_half_coef(xp_right)

    coords_left_x = np.concatenate([left_line_points[0], xp_left])
    coords_left_y = np.concatenate([left_line_points[1], left_line])

    coords_right_x = np.concatenate([right_line_points[0], xp_right])
    coords_right_y = np.concatenate([right_line_points[1], right_line])

    coefs_left = np.polyfit(coords_left_x, coords_left_y, deg=degree)
    coefs_right = np.polyfit(coords_right_x, coords_right_y, deg=degree)

    # xp = np.linspace(100, 800, 700)
    # left_poly, right_poly = np.poly1d(coefs_left), np.poly1d(coefs_right)
    # left_line = left_poly(xp)
    # right_line = right_poly(xp)
    #
    # plt.plot(xp, right_line)
    # plt.plot(xp, left_line)
    # plt.plot(left_line_points[0], left_line_points[1], 'r.')
    # plt.plot(right_line_points[0], right_line_points[1], 'g.')
    # plt.show()

    return coefs_left, coefs_right
