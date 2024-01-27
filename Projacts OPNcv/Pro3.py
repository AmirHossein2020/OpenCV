import cv2 
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:

    success, img = cap.read()

    imgRB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRB)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp.solutions.drawing_utils.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
        
        lmlist = []

        for id, lm in enumerate(hand.landmark):

            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmlist.append([id, cx, cy])
        
        if len(lmlist) != 0: 
            fingers = 0

            #Thumb
            if (lmlist[4][1] < lmlist[3][1]): fingers = fingers + 1

            #Index
            if (lmlist[8][2] < lmlist[7][2]): fingers = fingers + 1

            #Middle
            if (lmlist[12][2] < lmlist[11][2]): fingers = fingers + 1

            #Ring
            if (lmlist[16][2] < lmlist[15][2]): fingers = fingers + 1
            
            #pinke
            if (lmlist[20][2] < lmlist[19][2]): fingers = fingers + 1
                
        cv2.putText(img, f'{fingers}', (100, 300), cv2.FONT_HERSHEY_COMPLEX, 5, (0, 255, 0), 3)
    cv2.imshow("image", img)
    if (cv2.waitKey(1) & 0xFF == ord("q")):
            break