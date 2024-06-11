import os
import cv2
import numpy as np
def pituitary(): 
      print("hi")
      imagepath="uploads/upload_img.png"  
      Img = cv2.imread(imagepath)
      org = Img
      img = cv2.GaussianBlur(Img, (5,5), -1)  

      img_w_treshold = cv2.threshold(img,110, 150, cv2.ADAPTIVE_THRESH_MEAN_C)[1]
      cv2.imwrite('uploads/img_w_treshold.jpg', img_w_treshold)

      imagepath="uploads/img_w_treshold.jpg"
      image = cv2.imread(imagepath)

      img = cv2.GaussianBlur(image, (5,5), -1)
      Img = np.array(img)
      gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
      ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY  + cv2.THRESH_OTSU)


      kernel = np.ones((4, 4), np.uint8)
      opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
      curImg = opening
      sure_bg = cv2.dilate(curImg, kernel, iterations=3)
      dist_transform = cv2.distanceTransform(curImg, cv2.DIST_L2, 5)
      ret, sure_fg = cv2.threshold(dist_transform, 0.9* dist_transform.max(), 255, 0)
      sure_fg = np.uint8(sure_fg)
      unknown = cv2.subtract(sure_bg, sure_fg)
      ret, markers = cv2.connectedComponents(sure_fg)
      markers = markers + 1
      markers[unknown == 255] = 0
      if Img is not None:
            markers = cv2.watershed(Img, markers)
            Img[markers  == 1] = [0, 0, 0]
            cv2.imwrite('color.jpg', Img)
      else:
            print("Error: Unable to load the image.")

      tumorImage = cv2.cvtColor(Img, cv2.COLOR_BGR2HSV)

      curImg = tumorImage

      outputpath="uploads/seg23.png"

      cv2.imwrite(outputpath, curImg)
      out = cv2.imread(outputpath)
      img = cv2.addWeighted(org, 0.5, out, 0.9, 0)
      cv2.imwrite('uploads/imageseg.jpg', img)
      total_sum = np.sum(curImg)
      if total_sum<=605051:
            stage="first stage"
            return stage
      else:
            stage="second stage"
            return stage
            
