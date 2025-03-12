import numpy as np
import skimage
from skimage.io import imread,imsave,imshow
import cv2
# https://github.com/SartajBhuvaji/Brain-Tumor-Classification-Using-Deep-Learning-Algorithms/blob/master/Data%20Pre-Processing%20And%20Pickling/1_Data_Cropping%2BAugmentation.ipynb

def data_cropping(image):
    
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 30, 150)

    # enhance edges with morpho operations 
    kernel = np.ones((5,5))
    dilated_edges = cv2.dilate(edges, kernel, iteration = 2)
    eroded_edges = cv2.erode(dilated_edges, kernel)

    # find contours: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    contours, _ = cv2.findContours(eroded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check for the contours found by findContours
    #cv2.drawContours(image, contours, -1, (0,255,0), 3)

    # filter the contour found: if it is bigger than 1000 pixels (contour of the brain, we keep it)
    contours = [c for c in contours if cv2.contourArea(c) > 1000] 


    if contours:
        # Sort contours by area, keep the largest one to be sure that we don't loose any part of the brain
        contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # Find bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate the maximum dimension of the bounding box
        max_dim = max(w, h)

        # Calculate the coordinates of the square region centered around the brain
        center_x = x + w // 2
        center_y = y + h // 2
        x1 = max(0, center_x - max_dim // 2)
        y1 = max(0, center_y - max_dim // 2)
        x2 = min(image.shape[1], center_x + max_dim // 2)
        y2 = min(image.shape[0], center_y + max_dim // 2)

        # Crop the square region
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image
    else:
        print("No contours found")
        return None