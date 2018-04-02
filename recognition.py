# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:33:55 2018
@topic : Recognition for SAS
@author: rohit
"""
import numpy as np
import os
import cv2
import matplotlib.pyplot as plot
images = []
for file in os.listdir('processed_face_resize'):
    file_path = os.path.join('processed_face_resize',file)
    print('File Path : %s'%file_path)
    x = []
    x = cv2.imread(file_path)
    x = x[:,:,0]
    x = np.reshape(x,[60*60])
    images.append(x)

mean_image = np.zeros([3600]) 
mean_reduced_images = np.zeros([410,3600])
for i in range(len(images)):
    mean_image = mean_image + (images[i]/410)
mean_image_photo = np.reshape(mean_image,[60,60])
cv2.imwrite('Mean Face.jpg', mean_image_photo)
plot.imshow(mean_image_photo)
plot.show()
    
    

for i in range(len(images)):
   mean_reduced_images[i] = images[i] - mean_image

r = 25 #no.of pc -1
u,sigma,v = np.linalg.svd(mean_reduced_images)
red_dim = mean_reduced_images.dot(np.transpose(v[:r,:]))

photo = np.reshape(red_dim[10],[5,5])
plot.imshow(photo)
plot.show()
