from processing import *

# print(os.listdir("test_images/"))
# image = read_image('test_images/solidWhiteRight.jpg')
image = read_image('frame112.jpg')

#Convert yellow to white
image = yellow2white(image)

fig = plt.figure(figsize=(16, 8))
plt_size = [2, 4]

# Show original image
plt.subplot(plt_size[0], plt_size[1], 1)
plt.title('Original')
plt.imshow(image)

# Show black&white version
gray = grayscale(image)
plt.subplot(plt_size[0], plt_size[1], 2)
plt.title('Gray')
plt.imshow(gray, cmap='Greys_r')

# Show blurred image
blurred = gaussian_blur(gray, 3)
plt.subplot(plt_size[0], plt_size[1], 3)
plt.title('Blurred')
plt.imshow(blurred,cmap='Greys_r')


# Show with Canny applied
edges = canny(blurred, 50, 150)
plt.subplot(plt_size[0], plt_size[1], 4)
plt.title('Canny')
plt.imshow(edges,cmap='Greys_r')



# Show ROI
imshape = image.shape
vertices = np.array([[(imshape[1] * 0.12, imshape[0] * 0.9),  # left-bottom
                     (imshape[1] * 0.48, imshape[0] * 0.58),  # left-top
                     (imshape[1] * 0.55, imshape[0] * 0.58),  # right-top
                     (imshape[1] * 0.98, imshape[0] * 0.9)]], dtype=np.int32) # right-bottom
# vertices = np.array([[(imshape[1]*0.1,imshape[0]),
#                       (imshape[1]*0.45, imshape[0]*0.6),
#                       (imshape[1]*0.55, imshape[0]*0.6),
#                       (imshape[1],imshape[0])]], dtype=np.int32)
road_region_mask_test = region_of_interest(image, vertices)
plt.subplot(plt_size[0], plt_size[1], 5)
plt.title('ROI Original')
plt.imshow(road_region_mask_test)


# Show ROI on Canny filtered image
road_region_mask = region_of_interest(edges, vertices)
plt.subplot(plt_size[0], plt_size[1], 6)
plt.title('ROI Canny')
plt.imshow(road_region_mask,cmap='Greys_r')

# Draw Hough lines on the black background
lines_on_black = hough_lines(road_region_mask, 1, np.pi / 180, 30, 4, 2)
plt.subplot(plt_size[0],plt_size[1],7)
plt.title('Hough Lines')
plt.imshow(lines_on_black,cmap='Greys_r')


# Draw the lines on original image
resultant_img = weighted_img(lines_on_black, image)
plt.subplot(plt_size[0],plt_size[1],8)
plt.title('Hough Lines on scene')
plt.imshow(resultant_img)
plt.show()
