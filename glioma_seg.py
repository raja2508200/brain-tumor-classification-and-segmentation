import os
import cv2
import numpy as np

def gliomaseg():
    # Input image path
    imagepath = "uploads/upload_img.png"

    # Load the original image
    Img = cv2.imread(imagepath)
    org = Img

    # Apply GaussianBlur to the image
    img = cv2.GaussianBlur(Img, (5, 5), -1)

    # Apply threshold to the blurred image
    img_w_treshold = cv2.threshold(img, 95, 150, cv2.ADAPTIVE_THRESH_MEAN_C)[1]

    # Save the thresholded image
    cv2.imwrite('uploads/img_w_treshold.jpg', img_w_treshold)

    # Update the image path for the next operations
    imagepath = "uploads/img_w_treshold.jpg"
    image = cv2.imread(imagepath)

    # Apply GaussianBlur again
    img = cv2.GaussianBlur(image, (5, 5), -1)
    Img = np.array(img)
    gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 95, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((6, 6), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    curImg = opening
    sure_bg = cv2.dilate(curImg, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(curImg, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.9 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Check if the original image is not None before proceeding
    if Img is not None:
        # Apply watershed algorithm
        markers = cv2.watershed(Img, markers)
        Img[markers == 1] = [0, 0, 0]

        # Save the result of the watershed algorithm
        cv2.imwrite('uploads/color.jpg', Img)
    else:
        print("Error: Unable to load the image.")

    # Convert to HSV color space
    tumorImage = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)
    curImg = tumorImage

    # Save the segmented image
    outputpath = "uploads/seg23.png"
    cv2.imwrite(outputpath, curImg)

    # Read the saved segmented image
    out = cv2.imread(outputpath)

    # Blend the original image with the segmented image
    img = cv2.addWeighted(org, 0.5, out, 0.9, 0)

    # Save the final result
    cv2.imwrite('uploads/imageseg.jpg', img)

    # Calculate the area
    area = np.sum(curImg)

    # Determine the stage based on area
    if area <= 287948:
        stage = "first stage"
        return stage
    else:
        stage = "second stage"
        return stage


