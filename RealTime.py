import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
import efficientnet.tfkeras as efn
import pickle
import os
from gtts import gTTS

pickle_in = open("model.pck","rb")
dict_ = pickle.load(pickle_in)
dict_2 = {v: k for k, v in dict_.items()}

alpha = {0:'0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',10:'A',11: 'B',12: 'C',13: 'D',14: 'E',15: 'F',16: 'G',17: 'H',18: 'I',19: 'J',20: 'K',21: 'L',
22:'M',23: 'N',24: 'O',25: 'P',26: 'Q',27: 'R',28: 'S',29: 'T',30: 'U',31: 'V',32: 'W',33: 'X',34: 'Y',35: 'Z'}

model = load_model("FinalModel.h5")

from tensorflow.keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy',metrics = ['accuracy'])

def Process(img):
    img0 = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    ret , thresh = cv2.threshold(img0,30,255,cv2.THRESH_BINARY)
    img1[thresh != 0] = [255 , 255 , 255]
    img1 = cv2.resize(img1 , (224 , 224))
    return img1

def nothing(x):
    pass

background = None
accumulated_weight = 0.5
roi_top = 100
roi_bottom = 300
roi_right = 100
roi_left = 300

cv2.namedWindow('Camera Output')
cv2.createTrackbar('B for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for min', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('B for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('G for max', 'Camera Output', 0, 255, nothing)
cv2.createTrackbar('R for max', 'Camera Output', 0, 255, nothing)
     
cv2.setTrackbarPos('B for min', 'Camera Output', 0)
cv2.setTrackbarPos('G for min', 'Camera Output', 130)
cv2.setTrackbarPos('R for min', 'Camera Output', 103)
cv2.setTrackbarPos('B for max', 'Camera Output', 255)
cv2.setTrackbarPos('G for max', 'Camera Output', 182)
cv2.setTrackbarPos('R for max', 'Camera Output', 130)

#print("please")
def identify(img):
    img_copy = img.copy()
    img_copy = cv2.resize(img_copy , (224 , 224))
    img_copy = img_copy.reshape(1 , img_copy.shape[0] , img_copy.shape[1] , 3)
    img_copy = img_copy.astype(np.float16)
    pred = model.predict(img_copy)
    pred = np.argmax(pred)
    pred = dict_2[pred]
    return pred


def Get_frame():
    VideoCapture = cv2.VideoCapture(0)
    i =100
    prevCont = np.array([],dtype = np.int32)
    stat = 0
    detected = 0
    ans = ""
    while True :
        ret , frame = VideoCapture.read()
        roi = frame[100:300,100:300]
        cv2.rectangle(frame,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)
        
        imgray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thre = cv2.threshold(imgray, 127, 255, 0)
        
        imageYCrCb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)
        lower = np.array([cv2.getTrackbarPos('B for min', 'Camera Output'),
                             cv2.getTrackbarPos('G for min', 'Camera Output'),
                             cv2.getTrackbarPos('R for min', 'Camera Output')], np.uint8)
        upper= np.array([cv2.getTrackbarPos('B for max', 'Camera Output'),
                             cv2.getTrackbarPos('G for max', 'Camera Output'),
                             cv2.getTrackbarPos('R for max', 'Camera Output')], np.uint8)

        skinRegion = cv2.inRange(imageYCrCb, lower,upper)
        countors1,_ =cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours,_= cv2.findContours(thre, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #contours = sorted(contours, key=cv2.contourArea, reverse=True)
        match = cv2.matchShapes(contours[0],prevCont,2,0.0)
        prevCont = contours[0]
        
        stencil = np.zeros(roi.shape).astype(roi.dtype)
        color = [255, 255, 255]
        cv2.fillPoly(stencil,contours1, color)
        img= cv2.bitwise_and(roi, stencil)

        if(match > 0.70):
            stat = 0
        else:
            stat+=1 
        
        img_ = Process(img)
        
        if stat ==10:
            detected = 10
            pred = identify(img_)

        if detected > 0:
            if(pred != None):
                cv2.putText(roi, str(pred) , (100, 200), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
            detected-=1

        #if len(contours) ==  0:
         #  pred = "No Sign"
         key = cv2.waitKey(1)
        if key ==ord('y'):
            ans = ans + str(pred)
        if key == ord('\b'):
            ans = ans[:-1]
        if key == ord(' '):
            ans = ans + " "
        cv2.putText(frame, ans , (100, 250), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        cv2.drawContours(roi,contours,0,(0,255,0),1)

        cv2.imshow('Camera',frame)
        cv2.imshow('Background' , img)
        cv2.imshow('processed img' , img_)
        if key == ord('q'):
            break

    VideoCapture.release()
    cv2.destroyAllWindows()


speech = gTTS(ans)
speech.save("ans.mp3")
os.system("mpg123 ans.mp3")

Get_frame()


