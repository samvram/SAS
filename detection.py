import cv2 as cv
import os

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade.load('haarcascade_frontalface_default.xml')
# print(face_cascade)

images = []
for file in os.listdir('images'):
    file_path = os.path.join('images',file)
    print('File Path : %s'%file_path)
    x = dict()
    x['RGB'] = cv.imread(file_path)
    x['GRAY'] = cv.cvtColor(x['RGB'], cv.COLOR_BGR2GRAY)
    images.append(x)

print('Length of images : %s'%str(len(images)))
list_of_faces = []
for image in images:
    faces = face_cascade.detectMultiScale(image['GRAY'], 1.3, 5)
    for (x,y,w,h) in faces:
        image['RGB'] = cv.rectangle(image['RGB'],(x,y),(x+w,y+h),(255,0,0), 8)
        list_of_faces.append(image['RGB'][y:y+h, x:x+w])

i=0
for image in images:
    i = i + 1
    # print('image')
    cv.imwrite(os.path.join('processed_images','image_'+str(i)+'.jpg'), image['RGB'])

i=0
for face in list_of_faces:
    i = i + 1
    # print('faces')
    cv.imwrite(os.path.join('processed_faces','face_detailed_'+str(i)+'.jpg'), face)

print('End of Execution')

