from processing import *

DEBUG = False
prev_coeffs = None

def reset():
    global prev_coeffs
    prev_coeffs = [[None, None]] * 12

def ROI_coordinates(image):
    imshape = image.shape
    # Regular video params
    # return [[(imshape[1] * 0.05, imshape[0]),  # left-bottom
    #          (imshape[1] * 0.45, imshape[0] * 0.6),  # left-top
    #          (imshape[1] * 0.55, imshape[0] * 0.6),  # right-top
    #          (imshape[1] * 0.95, imshape[0])]]  # right-bottom

    # # Challenge video params
    return [[(imshape[1] * 0.15, imshape[0] * 0.92),  # left-bottom
             (imshape[1] * 0.45, imshape[0] * 0.6),  # left-top
             (imshape[1] * 0.55, imshape[0] * 0.6),  # right-top
             (imshape[1] * 0.95, imshape[0] * 0.92)]]  # right-bottom


def plot_results(image, gray, blurred, edges, road_region_mask_test, road_region_mask, lines_on_black, resultant_img):
    fig = plt.figure(figsize=(16, 8))
    plt_size = [2, 4]

    # Show original image
    plt.subplot(plt_size[0], plt_size[1], 1)
    plt.title('Original')
    plt.imshow(image)

    # Show black&white version
    plt.subplot(plt_size[0], plt_size[1], 2)
    plt.title('Gray')
    plt.imshow(gray, cmap='Greys_r')

    # Show blurred image
    plt.subplot(plt_size[0], plt_size[1], 3)
    plt.title('Blurred')
    plt.imshow(blurred, cmap='Greys_r')

    # Show with Canny applied
    plt.subplot(plt_size[0], plt_size[1], 4)
    plt.title('Canny')
    plt.imshow(edges, cmap='Greys_r')

    # Show ROI
    plt.subplot(plt_size[0], plt_size[1], 5)
    plt.title('ROI Original')
    plt.imshow(road_region_mask_test)

    # Show ROI on Canny filtered image
    plt.subplot(plt_size[0], plt_size[1], 6)
    plt.title('ROI Canny')
    plt.imshow(road_region_mask, cmap='Greys_r')

    # Draw Hough lines on the black background
    plt.subplot(plt_size[0], plt_size[1], 7)
    plt.title('Hough Lines')
    plt.imshow(lines_on_black, cmap='Greys_r')

    # Draw the lines on original image
    plt.subplot(plt_size[0], plt_size[1], 8)
    plt.title('Hough Lines on scene')
    plt.imshow(resultant_img)
    plt.show()


def process_image_hough(image, extrapolate=False):
    """
    Apply Canny and Hough Transform on the image to ROI on the image
    to extract the lane lines from the image
    :param image: RGB image array
    :return:
    """
    ROI = ROI_coordinates(image)
    gray = grayscale(image)
    blurred = gaussian_blur(gray, 3)
    edges = canny(blurred, 70, 190)
    vertices = np.array(ROI, dtype=np.int32)
    road_region_mask = region_of_interest(edges, vertices)
    lines_on_black = hough_lines(road_region_mask, 1, np.pi / 180, 30, 4, 2)
    resultant_img = weighted_img(lines_on_black, image)

    if DEBUG:
        road_region_mask_test = region_of_interest(image, vertices)
        plot_results(image,
                     gray,
                     blurred,
                     edges,
                     road_region_mask_test,
                     road_region_mask,
                     lines_on_black,
                     resultant_img)
    return resultant_img


def process_image_hough_extrapol(image):
    """
    Apply Canny and Hough Transform on the image to ROI on the image
    to extract the lane lines from the image and extrapolates them
    :param image: RGB image array
    :return:
    """
    ROI = ROI_coordinates(image)
    gray = grayscale(image)
    blurred = gaussian_blur(gray, 3)
    edges = canny(blurred, 70, 190)
    vertices = np.array(ROI, dtype=np.int32)
    road_region_mask = region_of_interest(edges, vertices)
    lines_on_black = hough_side_lines(road_region_mask, 1, np.pi / 180, 30, 4, 2)
    resultant_img = weighted_img(lines_on_black, image)

    if DEBUG:
        road_region_mask_test = region_of_interest(image, vertices)
        plot_results(image,
                     gray,
                     blurred,
                     edges,
                     road_region_mask_test,
                     road_region_mask,
                     lines_on_black,
                     resultant_img)
    return resultant_img


def process_image_hough_extrapol_poly(image):
    """
    Apply Canny and Hough Transform on the image to ROI on the image
    to extract the lane lines from the image and extrapolates them
    Uses prev_coeffs to smooth the lines (prevent jumps)
    :param image: RGB image array
    :return:
    """
    imshape = image.shape
    ROI = ROI_coordinates(image)
    gray = grayscale(image)
    blurred = gaussian_blur(gray, 3)
    edges = canny(blurred, 70, 190)
    vertices = np.array(ROI, dtype=np.int32)
    road_region_mask = region_of_interest(edges, vertices)

    rho, theta, threshold, min_line_len, max_line_gap = 1, np.pi / 180, 30, 4, 2
    lines = cv2.HoughLinesP(road_region_mask, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    left_coefs, right_coefs = extrapolate_poly(imshape, lines, degree=2)

    left_coefs = np.poly1d(left_coefs)
    right_coefs = np.poly1d(right_coefs)

    lines_on_black = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)
    draw_fitted_line(lines_on_black, [left_coefs, right_coefs], ROI, color=[255, 0, 0], thickness=12)

    resultant_img = weighted_img(lines_on_black, image)

    if DEBUG:
        road_region_mask_test = region_of_interest(image, vertices)
        plot_results(image,
                     gray,
                     blurred,
                     edges,
                     road_region_mask_test,
                     road_region_mask,
                     lines_on_black,
                     resultant_img)
    return resultant_img


def process_image_hough_extrapol_poly_mem(image):
    """
    Apply Canny and Hough Transform on the image to ROI on the image
    to extract the lane lines from the image and fits the polynom
    Uses previous coefficients to smooth the lines (prevent jumps)
    :param image: RGB image array
    :return:
    """
    global prev_coeffs
    imshape = image.shape
    ROI = ROI_coordinates(image)
    gray = grayscale(image)
    blurred = gaussian_blur(gray, 3)
    edges = canny(blurred, 70, 190)
    vertices = np.array(ROI, dtype=np.int32)
    road_region_mask = region_of_interest(edges, vertices)

    rho, theta, threshold, min_line_len, max_line_gap = 1, np.pi / 180, 30, 4, 2
    lines = cv2.HoughLinesP(road_region_mask, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    left_coefs, right_coefs = extrapolate_poly(imshape, lines, degree=2)
    # print(left_coefs)
    left_coefs, right_coefs = np.array([left_coefs]), np.array([right_coefs])
    for coeff in prev_coeffs:
        if coeff[0] is not None:
            left_coefs = np.concatenate((left_coefs, coeff[0].reshape(1, coeff[0].shape[0])), axis=0)
            right_coefs = np.concatenate((right_coefs, coeff[1].reshape(1, coeff[1].shape[0])), axis=0)
    left_coefs = left_coefs.mean(axis=0)
    right_coefs = right_coefs.mean(axis=0)
    prev_coeffs = prev_coeffs[1:]
    prev_coeffs.append([left_coefs, right_coefs])
    left_coefs = np.poly1d(left_coefs)
    right_coefs = np.poly1d(right_coefs)

    lines_on_black = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)
    draw_fitted_line(lines_on_black, [left_coefs, right_coefs], ROI, color=[0, 255, 0], thickness=5)

    resultant_img = weighted_img(lines_on_black, image)

    if DEBUG:
        road_region_mask_test = region_of_interest(image, vertices)
        plot_results(image,
                     gray,
                     blurred,
                     edges,
                     road_region_mask_test,
                     road_region_mask,
                     lines_on_black,
                     resultant_img)
    return resultant_img

def process_image_hough_extrapol_poly_mem2(image):
    """
    Apply Canny and Hough Transform on the image to ROI on the image
    to extract the lane lines from the image
    Uses previous coefficients to smooth the lines (prevent jumps)
    + uses the last bottom coordinates to make lines less bent to the sides
    :param image: RGB image array
    :return:
    """
    global prev_coeffs
    imshape = image.shape
    ROI = ROI_coordinates(image)
    gray = grayscale(image)
    blurred = gaussian_blur(gray, 3)
    edges = canny(blurred, 70, 190)
    vertices = np.array(ROI, dtype=np.int32)
    road_region_mask = region_of_interest(edges, vertices)

    rho, theta, threshold, min_line_len, max_line_gap = 1, np.pi / 180, 30, 4, 2
    lines = cv2.HoughLinesP(road_region_mask, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    left_coefs, right_coefs = extrapolate_poly_2(imshape, lines, degree=2)

    left_coefs, right_coefs = np.array([left_coefs]), np.array([right_coefs])
    for coeff in prev_coeffs:
        if coeff[0] is not None:
            left_coefs = np.concatenate((left_coefs, coeff[0].reshape(1, coeff[0].shape[0])), axis=0)
            right_coefs = np.concatenate((right_coefs, coeff[1].reshape(1, coeff[1].shape[0])), axis=0)

    # Average results
    left_coefs = left_coefs.mean(axis=0)
    right_coefs = right_coefs.mean(axis=0)

    #Save results
    prev_coeffs = prev_coeffs[1:]
    prev_coeffs.append([left_coefs, right_coefs])

    left_coefs = np.poly1d(left_coefs)
    right_coefs = np.poly1d(right_coefs)

    lines_on_black = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)
    draw_fitted_line(lines_on_black, [left_coefs, right_coefs], ROI, color=[0, 255, 0], thickness=5)

    resultant_img = weighted_img(lines_on_black, image)

    if DEBUG:
        road_region_mask_test = region_of_interest(image, vertices)
        plot_results(image,
                     gray,
                     blurred,
                     edges,
                     road_region_mask_test,
                     road_region_mask,
                     lines_on_black,
                     resultant_img)
    return resultant_img

def process_image_hough_extrapol_poly_mem2_yell(image):
    """
    Apply Canny and Hough Transform on the image to ROI on the image
    to extract the lane lines from the image
    Uses previous coefficients to smooth the lines (prevent jumps)
    + uses the last bottom coordinates to make lines less bent to the sides
    :param image: RGB image array
    :return:
    """
    global prev_coeffs
    processed = yellow2white(image)
    imshape = image.shape
    ROI = ROI_coordinates(image)
    gray = grayscale(processed)
    blurred = gaussian_blur(gray, 3)
    edges = canny(blurred, 50, 150)
    vertices = np.array(ROI, dtype=np.int32)
    road_region_mask = region_of_interest(edges, vertices)

    rho, theta, threshold, min_line_len, max_line_gap = 1, np.pi / 180, 35, 5, 2
    lines = cv2.HoughLinesP(road_region_mask, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    left_coefs, right_coefs = extrapolate_poly_2(imshape, lines, degree=2)

    left_coefs, right_coefs = np.array([left_coefs]), np.array([right_coefs])
    for coeff in prev_coeffs:
        if coeff[0] is not None:
            left_coefs = np.concatenate((left_coefs, coeff[0].reshape(1, coeff[0].shape[0])), axis=0)
            right_coefs = np.concatenate((right_coefs, coeff[1].reshape(1, coeff[1].shape[0])), axis=0)

    # Average results
    left_coefs = left_coefs.mean(axis=0)
    right_coefs = right_coefs.mean(axis=0)

    #Save results
    prev_coeffs = prev_coeffs[1:]
    prev_coeffs.append([left_coefs, right_coefs])

    left_coefs = np.poly1d(left_coefs)
    right_coefs = np.poly1d(right_coefs)

    lines_on_black = np.zeros((imshape[0], imshape[1], 3), dtype=np.uint8)
    draw_fitted_line(lines_on_black, [left_coefs, right_coefs], ROI, color=[255, 0, 0], thickness=5)

    resultant_img = weighted_img(lines_on_black, image)

    if DEBUG:
        road_region_mask_test = region_of_interest(image, vertices)
        plot_results(image,
                     gray,
                     blurred,
                     edges,
                     road_region_mask_test,
                     road_region_mask,
                     lines_on_black,
                     resultant_img)
    return resultant_img

if __name__ == "__main__":
    DEBUG = True
    # image = read_image('test_images/solidYellowLeft.jpg')
    image = read_image('frame112.jpg')
    imshape = image.shape
    # plt.imshow(process_image_hough(image))
    # process_image_hough_extrapol_poly(image)
    yellow2white(image)
