import cv2

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_result = cv2.adaptiveThreshold(img_result, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

    contours, _ = cv2.findContours(img_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    

    finalContours = []

    for c in contours:
        area = cv2.contourArea(c)

        if area > 2000:
            x, y , w , h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            finalContours.append(c)

    
    if success:

        height = int(cap.get(4))
        img = cv2.putText(img, str(len(finalContours)), (45, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5, cv2.LINE_AA)
        cv2.imshow("Image", img)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break