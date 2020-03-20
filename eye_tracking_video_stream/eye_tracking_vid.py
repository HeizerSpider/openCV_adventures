import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
    
    #eye detection
    # img = cv2.imread("frame",1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    path = "haarcascade_eye.xml"

    eye_cascade = cv2.CascadeClassifier(path)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.01,minNeighbors=20,minSize=(10,10))
    print(len(eyes))

    #draw out circles for all possible eyes
    for (x,y,w,h) in eyes:
        xc = x+w/2
        yc = y+h/2
        r = w/2
        cv2.circle(frame, (int(xc), int(yc)), int(r), (0,255,0), 2)

    cv2.imshow("Frame", frame)

    ch = cv2.waitKey(1) #run every 1 milisecond
    if ch & 0xFF == ord('q'): #letter to break out of the script
        break #to break out of the loop

cap.release 
cv2.detroyAllWindows()