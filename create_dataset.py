import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#detect the face of individuals

def face_extractor(img):
    #Now the 2nd step is to load the image and convert it into gray-scale.
   
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converts to grayscale

    faces=face_classifier.detectMultiScale(gray,1.3,5)#Parameters for detectMultiScale(gray scale variable,scaleFactor,minNeighbors)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_faces=img[y:y+h,x:x+w]


    return cropped_faces



cap=cv2.VideoCapture(0)
count=0

while True:
    ret,frame=cap.read()
    '''Extract face , convert to grayscale and save it in out folders'''
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
     
        file_name_path='C:/Users/HP/Desktop/project2/Face-Recognition-in-Live-Stream-using-VGG-16-main/data/dataset'+str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1)==13 or count==200:
        break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete!!")
import cv2
import numpy as np

face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#detect the face of individuals

def face_extractor(img):
    #Now the 2nd step is to load the image and convert it into gray-scale.
   
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#converts to grayscale

    faces=face_classifier.detectMultiScale(gray,1.3,5)#Parameters for detectMultiScale(gray scale variable,scaleFactor,minNeighbors)
    if faces is():
        return None
    for(x,y,w,h) in faces:
        cropped_faces=img[y:y+h,x:x+w]


    return cropped_faces



cap=cv2.VideoCapture(0)
count=0

while True:
    ret,frame=cap.read()
    '''Extract face , convert to grayscale and save it in out folders'''
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
     
        file_name_path='C:/Users/HP/Desktop/project2/Face-Recognition-in-Live-Stream-using-VGG-16-main/data/dataset'+str(count)+'.jpg'

        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not found")
        pass
    if cv2.waitKey(1)==13 or count==200:
        break
cap.release()
cv2.destroyAllWindows()
print("Collecting Samples Complete!!")
