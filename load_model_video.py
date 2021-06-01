import numpy as np
import cv2
import os

import faceRecognition as fr
print (fr)

# Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('out.avi', fourcc, 25, (1280, 720))

frame_number = 0
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'training-data\trainingData.yml')    #Give path of where trainingData.yml is saved

cap=cv2.VideoCapture("Office.mp4")   #If you want to recognise face from a video then replace 0 with video path
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

name={0:"John",1:"Rainn",2:"Steve"}    #Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.


while True:
    frame_number += 1
    ret,test_img=cap.read()

    if not ret:
        break

    faces_detected,gray_img=fr.faceDetection(test_img)

    print("face Detected: ",faces_detected)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)  
    
    
    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+h,x:x+h]
        label,confidence=face_recognizer.predict(roi_gray)
        print ("Confidence :",confidence)
        print("label :",label)
        fr.draw_rect(test_img,face)
        predicted_name=name[label]
        fr.put_text(test_img,predicted_name,x,y)
        
    #resized_img=cv2.resize(test_img,(1000,700))
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(test_img)
    

    #cv2.imshow("face detection ", resized_img)
    #cv2.imwrite('./gray/grayscale_%03d.jpg' % i, image_gray)
    #if cv2.waitKey(10)==ord('q'):
        #break

# All done!
cap.release()
cv2.destroyAllWindows()
