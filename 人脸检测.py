# -*- coding: utf-8 -*-
"""
Spyder Editor
需要在D盘先创建faceImagesGray\\yangdasen文件夹
This is a temporary script file.
"""

#import os
import cv2
from mtcnn.mtcnn import MTCNN

#outer_path = '/xuqiong/AgeGender/test_img_process/Qbbre'
#filelist = os.listdir(outer_path)  # 列举图片
 
detector = MTCNN()
k=0
for i in range(1,601):
    path="D:\\faceImages\\yangdasen\\%d.png" %(i)
    #src = os.path.join(os.path.abspath(outer_path), item)
    #cv2.namedWindow("Image")
    img = cv2.imread(path)
    #cv2.imshow("Image",img)
    #cv2.waitKey (0) 
    detected = detector.detect_faces(img)
    if len(detected) > 0:  # 大于0则检测到人脸
        for i, d in enumerate(detected):  # 单独框出每一张人脸
            x1, y1, w, h = d['box']
            x2 = x1 + w
            y2 = y1 + h
            image=img[(y1-10):(y2+10), (x1-10):(x2+10)]
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            pathgray="D:\\faceImagesGray\\yangdasen\\%d.png" %(k)
            cv2.imwrite(pathgray,gray_img)
            k=k+1
            
            #cv2.imwrite(src, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
  
    