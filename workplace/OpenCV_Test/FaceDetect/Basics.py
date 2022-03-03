# -*- coding = utf-8 -*-
# @Time: 2/7/2022 6:17 PM
# @Author: Li Daicheng
# @File: Basics.py
# @Software: PyCharm

import cv2
import numpy as np
import face_recognition

imgLYF = face_recognition.load_image_file("Image/1.webp")
imgLYF = cv2.cvtColor(imgLYF, cv2.COLOR_BGR2RGB)
imgLYFTest = face_recognition.load_image_file("Image/4.jpg")
imgLYFTest = cv2.cvtColor(imgLYFTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgLYF)[0]
encodeLYF = face_recognition.face_encodings(imgLYF)[0]
cv2.rectangle(imgLYF, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgLYFTest)[0]
encodeLYFTest = face_recognition.face_encodings(imgLYFTest)[0]
cv2.rectangle(imgLYFTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
results = face_recognition.compare_faces([encodeLYF], encodeLYFTest)
faceDis = face_recognition.face_distance([encodeLYF], encodeLYFTest)
print(results, faceDis)
cv2.putText(imgLYFTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Li Yifeng", imgLYF)
cv2.imshow('Li Yifeng Test', imgLYFTest)
cv2.waitKey(0)
