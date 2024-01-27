import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import random

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

timer = 0
stateResalt = False
startGame = False
scoress = [0,0]


while True:
    imgBG = cv2.imread("Images/BG.png")
    success, img = cap.read()

    imgScaled = cv2.resize(img,(0,0),None, 0.393, 0.393)
    imgScaled = imgScaled[:,0:180]

     # find hands
    hands, img = detector.findHands(imgScaled)

    if startGame:
        
        if stateResalt is False:
            timer = time.time() - intialTime
            cv2.putText(imgBG, str(int(timer)), (260,205), cv2.FONT_HERSHEY_PLAIN, 4,(255,0,255),4)

            if timer > 3:
                stateResalt = True
                timer = 0

                if hands:
                    hand = hands[0]
                    playerMove = 0
                    fingers = detector.fingersUp(hand)
                    if fingers == [0,0,0,0,0]:
                        playerMove = 1
                    if fingers == [1,1,1,1,1]:
                        playerMove = 2
                    if fingers == [0,1,1,0,0]:
                        playerMove = 3

                    rendomNamber = random.randint(1, 3)
                    imgAI = cv2.imread(f"Images/{rendomNamber}.png",cv2.IMREAD_UNCHANGED)
                    
                    if (playerMove == 1 and rendomNamber == 3) or \
                            (playerMove == 2 and rendomNamber == 1) or \
                            (playerMove == 3 and rendomNamber == 2):
                          scoress[1] += 1 
                    if (playerMove == 3 and rendomNamber == 1) or \
                            (playerMove == 1 and rendomNamber == 2) or \
                            (playerMove == 2 and rendomNamber == 3):
                          scoress[0] += 1 
                    if scoress[1] or scoress[0] == 5:
                        break
    
                    print(playerMove)
    
    imgBG[106:295,355:535] = imgScaled

    if stateResalt:
        imgBG = cvzone.overlayPNG(imgBG, imgAI,(-12,70))

    cv2.putText(imgBG, str(scoress[0]), (180,98), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),4)
    cv2.putText(imgBG, str(scoress[1]), (500,98), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255),4)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
   # cv2.imshow("Image", img)
    cv2.imshow("BG", imgBG)
   # cv2.imshow("Scaled", imgScaled)
    
    key = cv2.waitKey(1)
    if key == ord("s"):
        startGame = True
        intialTime = time.time()
        stateResalt = False
    

    