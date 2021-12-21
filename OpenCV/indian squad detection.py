Original file is located at
    https://colab.research.google.com/drive/1HK40O72-pJKT4unhLA2XjDgjGfk-Tqdo
"""

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('squad.jpg',1)
print(type(img))

plt.imshow(img) # converting BGR to RGB for using matplotlib
plt.axis('off')
plt.show()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img) # converting BGR to RGB for using matplotlib
plt.axis('off')
plt.show()

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap = 'gray')
plt.axis('off')
plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

for (x,y,w,h) in faces:
    img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
    
print(len(faces))

img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2_imshow(img)
