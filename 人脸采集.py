# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:46:54 2019

@author: Administrator
"""
from tkinter import *
import cv2
from PIL import Image, ImageTk#图像控件
import threading#多线程 

def cc():
        while True:
            ret, frame = capture.read()#从摄像头读取照片
            frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file=ImageTk.PhotoImage(img)
            canvas.create_image(0,0,anchor='nw',image=image_file)
            
def mkdir(path):
    # 引入模块
    import os 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\") 
    # 判断路径是否存在 存在True 不存在False
    isExists=os.path.exists(path) 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        entry1.delete(0, END)
        entry1.insert(index=0,string='创建文件夹成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        #entry1.select_clear()
        entry1.delete(0, END)  # 删除所有值
        entry1.insert(index=0,string='目录已存在')
        #print(path+' 目录已存在')
        return False 
    
def build():
    name=entry.get()
    name=name.strip()
    if name=='':
        entry1.delete(0, END)
        entry1.insert(index=0,string='请输入名字')
    else:
        mkpath="D:\\faceImages\\%s \\" %(name)
        # 调用函数
        mkdir(mkpath)
        
def video_demo():
    global capture
    capture = cv2.VideoCapture(0)  
    t=threading.Thread(target=cc)
    t.start()


        
            
def save():
    def cc1():
        for i in range(1,601):
            ret, frame = capture.read()#从摄像头读取照片
            path="D:\\faceImages\\%s\\%d.png" %(name,i)
            cv2.imwrite(path,frame)
            entry2.delete(0, END)
            entry2.insert(index=0,string='%d'%(i))
            frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file=ImageTk.PhotoImage(img)
            canvas.create_image(0,0,anchor='nw',image=image_file)
            #print(path)
    name=entry.get()
    name=name.strip()
    if name=='':
        entry1.delete(0, END)
        entry1.insert(index=0,string='请输入名字')
    else:
        t=threading.Thread(target=cc1)
        t.start()
        #for i in range(1,601):
            #sucess,img = capture.read()
            #path="E:\\faceImages\\%s\\%d.png" %(name,i)
           # print(path)
            #cv2.imwrite(path,img)
            #entry2.delete(0, END)
           # entry2.insert(index=0,string='%d'%(i))
            
def Exit():
      capture.release()
      window.destroy()

        


window=Tk()
# 第2步，给窗口的可视化起名字
window.title('人脸采集')
# 第3步，设定窗口的大小(长 * 宽)
window.geometry('600x600+700+70')  # 这里的乘是小x
l=Label(window, text='步骤一：请输入你的名字，然后点击确认：',font=('华文行楷',13))
l.grid()    # Label内容content区域放置位置，自动调节尺寸
entry=Entry(window,font=('华文行楷',15),width=15)
entry.grid(row=0,column=1,sticky=W)
button=Button(window,text='确认',font=('华文行楷',15),command=build)
button.grid(row=0,column=2)

entry1=Entry(window,font=('华文行楷',10))
entry1.grid(row=1,column=1)
        
button1=Button(window,text='三:开始保存照片',font=('华文行楷',15),command=save)
button1.grid(row=2,column=1)

button2=Button(window,text='步骤二:打开摄像头',font=('华文行楷',15),command=video_demo)
button2.grid(row=2,column=0)

canvas = Canvas(window,height=500, width=470,bg='black')
canvas.grid(row=3,column=0,columnspan=2,rowspan=20)

label2=Label(window, text='已保存：',font=('华文行楷',15))
label2.grid(row=3,column=2,sticky=N)

entry2=Entry(window,font=('华文行楷'),width=10)
entry2.grid(row=4,column=2,sticky=N)

button4=Button(window,text='退出',font=('华文行楷',15),command=Exit)
button4.grid(row=20,column=2)   
window.mainloop()





