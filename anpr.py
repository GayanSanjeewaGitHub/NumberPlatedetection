import threading
import cv2
from cv2 import dnn_superres
import numpy as np
import datetime
from transform import four_point_transform
import re
import copy
#from lpfix import lpfix

from PIL import Image
import requests
from io import BytesIO

import sys
print("Python:", sys.version)
print("CV2:   ", cv2.__version__)

from pathlib import Path

# Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
sr = cv2.dnn_superres.DnnSuperResImpl_create() 
path = "./dnn_superres/ESPCN_x4.pb" 
sr.readModel(path)
sr.setModel("espcn", 4) # set the model by passing the value and the upsampling ratio

# multilingual OCR toolkits based on PaddlePaddle
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang='en', show_log=False) # need to run only once to download and load model into memory

# YOLOv5 is a family of object detection architectures and models 
import torch
model = torch.hub.load('./yolov5/', 'custom', path='./weight/best.pt', source='local')

# license plate blacklist 
import blacklist
from blacklist import search

# change to false if have GPU for testing
optimization = True

streams=( # set links for ip cams and different names for each cam
    ['./video/1.mp4','video 1'],
    ['./video/2.mp4','video 2'],
    ['./video/3.mp4','video 3'],
    )

# font and colors for img info
fontface = cv2.FONT_HERSHEY_DUPLEX 
colortxt=(255,255,255) # frames ocr text color
colorlp=(200,100,30) # frames lp outline and ocr text background color
tl=3
tf = max(tl - 2, 1)  # font thickness

def cams(s): # read frames from video files and save to list

    cam=s[1]
    cap=cv2.VideoCapture(s[0])
    fps=cap.get(cv2.CAP_PROP_FPS)
    fps = round(fps, 0)
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print ('Connected IP cam '+cam+' '+str(fps)+ ' FPS')
    framecounter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame is not None:
                now = datetime.datetime.now()
                framecounter = framecounter + 1
                img = frame
                
                if optimization:
                    # fps optimization
                    if fps > 45:
                        if (framecounter % 25 == 0):
                            frames.append([now,cam,img])
                    elif fps > 25:
                        if (framecounter % 15 == 0):
                            frames.append([now,cam,img])
                    elif fps > 10:
                        if (framecounter % 5 == 0):
                            frames.append([now,cam,img])
                    else:
                        frames.append([now,cam,img])
                else:
                    frames.append([now,cam,img])

    cap.release()

def frame():
    while True:

        if len(frames)>0: # if no frames skip detection
            
            now=frames[0][0]
            cam=frames[0][1]
            img=frames[0][2]

            plate = model(img)


            plate = plate.pandas().xyxy[0].to_dict(orient='records')


            for lp in plate:
                
                con = int(lp['confidence']*100) # confidence in %
                cs = lp['class'] # only one lp class in this weight
                x1 = int(lp['xmin'])
                y1 = int(lp['ymin'])
                x2 = int(lp['xmax'])
                y2 = int(lp['ymax'])

                h=y2-y1 # detected box height
                w=x2-x1 # detected box width
                ratio = float(w) / float(h) # detected box ratio

                if con > 50 and h > 15 and w > 50 and ratio > 1.5 and ratio <7: # yolov5s detection rules
                    
                    lproi = img[y1:y1+h, x1:x1+w] # crop lproi for saving to folder
                    lproi = cv2.cvtColor(lproi, cv2.COLOR_BGR2GRAY) # lproi to gray
                    lproiorig = lproi # lproi copy
                    lproi = sr.upsample(lproi) # dnn superres lproi upscale
                    
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4)) # Contrast Limited Adaptive Histogram Equalization
                    lproi = clahe.apply(lproi)
                    lproi2 = lproi

                    c1, c2 = ((x1, y1), (x2,y2))
                    cv2.rectangle(img, c1, c2, colorlp, thickness=tl, lineType=cv2.LINE_AA)  # draw LP outline rectangle
                    
                    det = ocr.ocr(lproi,rec=False,cls=False) # paddle ocr text detection without recognition
                    #top-left, top-right, bottom-right, bottom-left
                    #[[[[55.0, 38.0], [730.0, 14.0], [736.0, 179.0], [60.0, 203.0]]]]
                    #not perform recognition and classification tasks respectively

                    text=""
                    textsub=""
                    for d in det: # for each paddleocr text detection
                         #print([x - 5 for x in d[0][3]])
                        try: # expand detected text box
                            #corresponds to the left top corner of the box, and the subsequent points go in clockwise order around the box
                            pts = np.array([(d[0][0][0]-10, d[0][0][1]+10), (d[0][1][0]+10,  d[0][1][1]+10), (d[0][2][0]+10, d[0][2][1]-10), (d[0][3][0]-10, d[0][3][1]-10)], dtype = "int") # box detection points
                            #pts = np.array([(int(d[0][0])-10, int(d[0][1])-10), (int(d[1][0])+10, int(d[1][1])-10), (int(d[2][0])+10, int(d[2][1])+10), (int(d[3][0])-10, int(d[3][1])+10)], dtype = "int") # box detection points
                        except: # original detected text box
                            pts = np.array([(d[0][0][0], d[0][0][1]), (d[0][1][0], d[0][1][1]), (d[0][2][0], d[0][2][1]), (d[0][3][0], d[0][3][1])], dtype = "int") # box detection points
                            #pts = np.array([(d[0][0], d[0][1]), (d[1][0], d[1][1]), (d[2][0], d[2][1]), (d[3][0], d[3][1])], dtype = "int") # box detection points

                        lproi = four_point_transform(lproi2, pts) # perspective transform
                        #print("--------------------------------------------------------------------------------")


                        rec = ocr.ocr(lproi,det=False,cls=False) # ocr text recognition

                        for r in rec: # for each paddleocr text recognition
                                                    
                            scores = (rec[0][0][1])
                            if np.isnan(scores):
                                scores = 0
                            else:
                                scores = int(scores*100)
                            
                            if scores>90:
                                textsub = (rec[0][0][0])
                                text=text+textsub

                    pattern = re.compile('[\W]')
                    text=pattern.sub('', text)
                    text=text.replace("O","0")
                    text=text.replace("_","")

                    timestampcheck = now.strftime('%Y%m%d%H%M') # timestamp for duplicate LP check

                    original2 = copy.deepcopy(original)
                    newlp = False
                    
                    if (len(text)>3) and (len(text)<10): # ocr recognition rules

                        for dict_cam, dict_time in original2.items(): # duplicate LP check

                            if len(original[dict_cam])<1 and str(dict_cam) == str(cam):
                                original[cam][text] = timestampcheck
                                newlp = True
                            else:
                                for dict_lp in dict_time:

                                    if str(dict_cam) == str(cam):                                                         
                                        # change time for old LP
                                        if str(dict_lp) == str(text):
                                            original[cam][text] = timestampcheck
                                            newlp = False
                                            break
                                        # add new LP
                                        elif str(dict_lp) != str(text):
                                            original[cam][text] = timestampcheck
                                            newlp = True
                            
                                    # delete old LP
                                    if int(dict_time[dict_lp])<int(timestampcheck):
                                        del original[dict_cam][dict_lp]

                            # lp info on img
                            t_size = cv2.getTextSize(text, 0, fontScale=tl / 3, thickness=tf)[0]
                            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                            cv2.rectangle(img, c1, c2, colorlp, thickness=-1, lineType=cv2.LINE_AA)  # draw rectangle above lp for ocr text
                            cv2.putText(img, text, (c1[0], c1[1] - 2), fontface, tl / 3, colortxt, thickness=tf, lineType=cv2.LINE_AA) # draw ocr text in rectangle                     

                        if newlp: # if new LP is detected
                            
                            timestamp = now.strftime('%Y%m%d%H%M%S%f')[:-6] # timestamp for lproi file name and cam img preview

                            folder = now.strftime('./lp/%Y/%m/%d/%H/%M/') # folder for write lproi
                            Path(str(folder)).mkdir(parents=True, exist_ok=True) # if folder and subfolder not exist create

                            lproi = cv2.resize(lproi,(105,22), interpolation=cv2.INTER_LINEAR) # resize all lproi at same size
                            cv2.imwrite(str(folder)+str(timestamp)+'-'+str(cam)+'-'+str(text)+'.jpg',lproiorig) # write lproi to folder

                            print (timestamp,' ',cam,' ',text) # print data to console
                                                        
                            search(img,lproi,cam,now,text) # blacklist check

            # date and video name to img
            camdate = now.strftime('%d.%m.%Y %H:%M:%S')
            cv2.putText(img, str(camdate), (10, 20), fontface, 0.5, colortxt, thickness=tf, lineType=cv2.LINE_AA) # draw timestamp to img
            cv2.putText(img, str(cam), (10, 40), fontface, 0.5, colortxt, thickness=tf, lineType=cv2.LINE_AA) # draw cam name to img

            scale_percent = 50 # percent of original img dimensions for preview many ip cams on screen
            
            # calculate the percent of original dimensions
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dsize = (width, height) # dsize         
            img = cv2.resize(img, dsize) # resize preview img
            
            cv2.waitKey(1)
            cv2.imshow(cam, img) # show img with data

            del frames[0] # delete detected frame

if __name__ == "__main__":

    original={}

    for s in streams:
        original[s[1]] = {}

    thread_list = []
    frames = []
    
    for s in streams:
        x = threading.Thread(target = cams,args = (s,))
        x.start()
          
    y = threading.Thread(target = frame)
    y.start()
    
    print('Active threads - ', threading.activeCount())