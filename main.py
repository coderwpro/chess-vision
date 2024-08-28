import cv2
import imutils 
from imutils import paths

cap=cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey,(5, 5), 0)
    edged = cv2.Canny(grey, 35, 125)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key= cv2.contourArea)
    cv2.imshow("video", grey)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()

