import cv2
import numpy as np

# Load the image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Define Prewitt kernel
kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Apply convolution using filter2D function
img_x = cv2.filter2D(img, -1, kernel_x)
img_y = cv2.filter2D(img, -1, kernel_y)

# Combine the gradient images to get the magnitude
magnitude = np.sqrt(img_x**2 + img_y**2).astype(np.uint8)

# Normalize the magnitude image
cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)

# Display the original image and the filtered image
cv2.imshow('Original Image', img)
cv2.imshow('Prewitt Filtered Image', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()
