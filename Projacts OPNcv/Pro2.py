import cv2 
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480)

decetctor = FaceDetector(minDetectionCon=0.75)

while True:
    success, img = cap.read()

    if success:
        img, boxs = decetctor.findFaces(img, draw=True)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        cv2.imshow("Image", img)
cv2.destroyAllWindows()