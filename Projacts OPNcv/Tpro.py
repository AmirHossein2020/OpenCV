import cv2
import numpy as np
import face_recognition


imgAmi = face_recognition.load_image_file('Images2/PXL_20230806_180150770~2.jpg')
imgAmi = cv2.cvtColor(imgAmi,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Images2/PXL_20230806_180150770~2.jpg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)




faceLoc = face_recognition.face_locations(imgAmi)[0]
encodeAmir = face_recognition.face_encodings(imgAmi)[0]
cv2.rectangle(imgAmi,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLoc[3],faceLoc[0],faceLoc[1],faceLoc[2]),(255,0,255),2)


results = face_recognition.compare_faces([encodeAmir],encodeTest)
faceDis = face_recognition.face_distance([encodeAmir],encodeTest)
print(results)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Amir",imgAmi)
cv2.imshow("Test",imgTest)
cv2.waitKey(0)
