import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

img = cv2.imread('C:/data/elon_musk.jpg') # change the path to the directory that has you image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting the image to grayscale

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # creating a rectangular box around the face detected
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)  # creating a rectangular box around the eyes detected

cv2.imshow('img',img) # this is the resultant
cv2.waitKey(0)
cv2.destroyAllWindows()