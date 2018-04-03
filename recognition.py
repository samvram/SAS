# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:33:55 2018
@topic : Recognition for SAS
@author: rohit
"""
import numpy as np
import os
import cv2
from preproc import stretch
from sklearn.linear_model import LogisticRegression
from PIL import Image

images = []
labels = []
with open('train/label.txt','r') as f:
    str = f.read()
    for line in str.split('\n'):
        labels.append(line.split(' ')[1])
        file = 'face_detailed_'+line.split()[0]+'.jpg'
        file_path = os.path.join('train',file)
        print('File Path : %s'%file_path)
        # x = []
        x = cv2.imread(file_path)
        x = x[:,:,0]
        x = np.reshape(x,[60*60])
        images.append(x)

mean_image = np.zeros([3600]) 
mean_reduced_images = np.zeros([410,3600])
for i in range(len(images)):
    mean_image = mean_image + images[i]
mean_image = mean_image/len(images)
mean_image_photo = np.reshape(mean_image,[60,60])
cv2.imwrite('Mean Face.jpg', mean_image_photo)
# plot.imshow(mean_image_photo)
# plot.show()
    
    

for i in range(len(images)):
   mean_reduced_images[i] = images[i] - mean_image

r = 25 #no.of pc -1
u,sigma,v = np.linalg.svd(mean_reduced_images)
red_dim = mean_reduced_images.dot(np.transpose(v[:r,:]))

# photo = np.reshape(red_dim[10],[5,5])
# plot.imshow(photo)
# plot.show()

logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(red_dim, labels)

# file_path = input('Enter the File Path of today\'s picture: ')
file_path = os.path.join('test','test.jpg')

image = cv2.imread(file_path)
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade_alt2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

list_of_faces = []
faces = face_cascade_alt2.detectMultiScale(img, 1.3, 5)
for (x,y,w,h) in faces:
    # cv.rectangle(image['GRAY'],(x,y),(x+w,y+h),(255,0,0), 8)
    list_of_faces.append(stretch(img[y:y+h, x:x+w]))
faces_2 = face_cascade.detectMultiScale(img, 1.3, 5)
for (x,y,w,h) in faces_2:
    if (x,y,w,h) not in faces:
        # cv.rectangle(image['GRAY'], (x, y), (x + w, y + h), (255, 0, 0), 8)
        list_of_faces.append(stretch(img[y:y + h, x:x + w]))

i=0
for face in list_of_faces:
    i = i + 1
    cv2.imwrite(os.path.join('buffer','buffer_%d.jpg'%i), face)

list_of_faces = []
for file in os.listdir('buffer'):
    img = Image.open(os.path.join('buffer',file))
    img = img.resize((60,60), Image.ANTIALIAS)
    list_of_faces.append(img)

Attendance = dict()

for label in labels:
    Attendance[label] = 'Absent'

for face in list_of_faces:
    f = np.reshape(face,[3600,])
    f = f - mean_image
    X = f.dot(np.transpose(v[:r,:]))
    print(logistic_regression_model.predict([X]))
    Attendance[logistic_regression_model.predict(X)[0]] = 'Present'

for label,atd in Attendance.items():
    print('%s - %s'%(label,atd))