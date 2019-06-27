# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:22:16 2019

@author: Saba

Video Online face Detection:
"""

import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap= cv2.VideoCapture(0)

while True:
    ret, img = cap.read()     #output video, and second that reading is successful
   # if ret == False:
    #    break
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_img, 0)


    
    for face in rects:
        x_list = []
        y_list = []
        (x1, y1) = (face.left(), face.top())
        (x2, y2) = (face.right(), face.bottom())
        cv2.rectangle (img,(x1, y1), (x2, y2), (0, 255, 0), 3 )
        shape = predictor(gray_img, face)
        shape = face_utils.shape_to_np(shape)
        # find center of the face:
        mean =  np.mean(shape, axis = 0) # find mean on x axis
        meanx, meany = int(mean[0]), int(mean[1])
        
        
        
        # shape is not numpy and need to be numpy!
        #for i in shape:
           # x_list.append(shape.part(i).x)
           # y_list.append(shape.part(i).y)
        for (x, y) in shape:
            cv2.circle(img, (x, y), 3, (0 , 0, 255), -1)
            cv2.circle(img, (meanx, meany), 3, (255, 0, 0), -1)
            cv2.line(img, (meanx, meany), (x, y), (0,255,0), 1)
        
    
    cv2.imshow("face", img)
    

    cv2.waitKey(5) # delay in showing
    
    if cv2.waitKey(30) == ord('q'):   
        break
cap.release()
cv2.destroyAllWindows()
