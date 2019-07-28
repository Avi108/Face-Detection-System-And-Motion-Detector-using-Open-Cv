pip install opencv-python
$ pip install opencv-python
import cv2
import os
import numpy as np


img = cv2.imread("C:\\Resume and CV\\10.jpg",1)
print(img)
print(type(img))
print(img.shape)

img2= cv2.imread("C:\\Resume and CV\\K AVINASH REDDY.jpg",0)
print(img2)
print(type(img2))
print(img2.shape)

cv2.imshow("Rio",img)
cv2.waitKey(200)

cv2.destoryAllwindows()


resize = cv2.resize(img,(600,600))
print(resize.shape)
cv2.imshow("Rio1",resize)
cv2.waitKey(200)

#create  a casscase classifier  - object  creater a filter 
face_cascade = cv2.CascadeClassifier("C:\\OpenCV\\haarcascade_frontalface_default.xml")
#reading the image

img = cv2.imread("C:\\Resume and CV\\10.jpg",1)

#converting it into a grey scale image
 
grey_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#search for the co-oridnates of the image
faces = face_cascade.detectMultiScale(grey_img,scaleFactor = 1.05, minNeighbors = 5)

print(type(faces))
print(faces)
#this is a method to search for face recentage co-ordinates and scale factors dec shape by 5% smaller the value greater the accuracy

for x,y,w,h in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    
    #method to create a face rectangel 
    #r g b value -- co-ordinates 
    #3 is width of rectange
 #resize the image

resize = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))   

cv2.imshow("array",resize)
cv2.waitKey(0)
cv2.destoryAllwindows()
#creates a cascade 

#symmentrical
re = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))

cv2.waitKey(200)
cv2.destoryAllwindows()


#capturing Video 
import cv2,time
video = cv2.VideoCapture(0)
time.sleep(3)
#required a time module 
check,frame = video.read()
#check returns True or False and Frame is a n dimensional array
print(check)
print(frame)
 #captures the 1st frame of the video
cv2.imshow("capture",frame)
cv2.waitKey(0)

video.release()
cv2.destoryAllWindows()
#video capture object - Returns T or F to show video capture object

#capturing full video with multiple frames using while loop
import cv2,time
video3 = cv2.VideoCapture(0)
a= 1 #while the python is able to read the video capture object execute the loop
while True: #iterate frames and display the video
    a = a +1
    check,frame = video.read()
    print(frame)
    print(check)
    gray = cv2.cvtColor(frame,cv2.COLORBGR2GRAY) #Convert each frame in gray scale
    cv2.imshow('Capture',gray)
    key = cv2.waitKey(1) #new frame after every 1 milliselecond -- wait key
    if key == ord('q'):    #once we enter q this break statment will break the loop 
        break
 
    
 print(a)  #this is the number of frames 
 video.release()
 cv2.destoryAllWindow()
    
    
    

import cv2,time
video1 = cv2.VideoCapture("C:\\OpenCV\\VID20190613130029.mp4")
 check1,frame1 = video1.read()
#check returns True or False and Frame is a n dimensional array
print(check1)
print(frame1)

cv2.imshow("capture",frame1)
cv2.waitKey(0)

video.release()
cv2.destoryAllWindows()

#Motion Dector
import cv2,time
first_frame = None
video = cv2.VideoCapture(0)  #creates a videocapture obj 
while True:
    check,frame = video.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #convert the frame to gray scale
    gray= cv2.GaussianBlur(gray,(21,21),0) # convet the gray scale to Gaussian Blur image
    if first_frame in None: #store my 1st frame variable 
        first_frame = gray
        continue#come out 
    #gray is the subsequent frame
    delta = cv2.absdiff(first_frame,gray) #cal the diff 
    threst_delta = cv2.threshold(delta,30,255,cv2.THRESH_BINARY)[1] #define threshold it will convert the diff value with lessthan 30 to black .if >30 than willconver to white
    threst_delta = cv2.dilate(thresh_delta,None,iterations=0) #gray+ white
    (_,cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #defien the borders it will keep only that part of image greater than 1000 pixel removes unwanterd images 
    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue 
        (x,y,w,h)= cv2.boudingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow('frame',frame)
        cv2.imshow('Caputure',gray)
        cv2.imshow('delta',delta_frame)
        cv2.imshow('thresh',threst_delta)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
video.release()
cv2.destoryAllWindows()
#calculate the Time for which object is in front of the image
first_frame = None
status_list= [None,None]
times = []
df = pandas.DataFrame(columns=["Start","End"]) #when object came in front of camera and when object went off
video = cv2.VideoCapture(0)
while True:
    check,frame = video.read()
    status= 0                  #break the recording as the object is no avaliable we use status
    gray = cv2.cvtColor(frames,cv2.COLORBGR2GRAY)
    gray= cv2.TGaussianBlur(gray,(21,21),0)
#wil change status when object is detected
      (_,cnts,_) = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) 
      for contour in cnts:
          if cv2.contourArea(contour) < 1000:
              continue
          status = 1
          (x,y,w,h) = cv2.boundingRect(contour)
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
#list of status for every frame
status_list.append(status)
status_list = status_list[-2:] #last but one frame
#records time  if status -- list and 2nd last obj in previious frame 
#my last frame had  no obj = 2nd last frame had ab object 
if status_list[-1] == 1 and status_list[-2] == 0:
    time.append(datetime.now())
if status_list[-1] == 0 and status_list[-2] == 1:
    time.append(datetime.now())
    
print(status_list)
print(times)
for i in range(0,len(times),2):
    #stores time values in a data frame
    df = df.append({"Start":times[i],"End":times[i+1]},ignore index= TRUE)
df.to_csv("Times.csv")
#write the time frame
video.release()
cv2.destoryAllWindows()
    
