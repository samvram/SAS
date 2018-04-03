import cv2 as cv
import os
from preproc import stretch
from PIL import Image

face_cascade_alt2 = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
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
    x['PAR'] = x['GRAY']
    x['RGB'] = x['GRAY']
    images.append(x)

print('Length of images : %s'%str(len(images)))
list_of_faces = []
for image in images:
    faces = face_cascade_alt2.detectMultiScale(image['GRAY'], 1.3, 5)
    for (x,y,w,h) in faces:
        # cv.rectangle(image['GRAY'],(x,y),(x+w,y+h),(255,0,0), 8)
        list_of_faces.append(stretch(image['RGB'][y:y+h, x:x+w]))
    faces_2 = face_cascade.detectMultiScale(image['GRAY'], 1.3, 5)
    for (x,y,w,h) in faces_2:
        if (x,y,w,h) not in faces:
            # cv.rectangle(image['GRAY'], (x, y), (x + w, y + h), (255, 0, 0), 8)
            list_of_faces.append(stretch(image['RGB'][y:y + h, x:x + w]))

i=0
for file in os.listdir('processed_images'):
    os.remove(os.path.join('processed_images',file))

for image in images:
    i = i + 1
    # print('image')
    cv.imwrite(os.path.join('processed_images','image_'+str(i)+'.jpg'), image['RGB'])

i=0
for file in os.listdir('processed_faces'):
    os.remove(os.path.join('processed_faces',file))

for face in list_of_faces:
    i = i + 1
    # print('faces')
    cv.imwrite(os.path.join('processed_faces','face_detailed_'+str(i)+'.jpg'), face)

for file in os.listdir('processed_face_resize'):
    os.remove(os.path.join('processed_face_resize',file))

for file in os.listdir('processed_faces'):
    img = Image.open(os.path.join('processed_faces',file))
    img = img.resize((60,60), Image.ANTIALIAS)
    img.save(os.path.join('processed_face_resize',file))


print('End of Execution')

