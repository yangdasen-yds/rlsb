# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:25:32 2019
#需要和faceImagesGray与cnn文件夹放在同一个文件夹下
@author: Administrator
"""
from __future__ import print_function
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import numpy as np,numpy
import os
from PIL import Image
#import tensorflow as tf

num_dirs = 0 #路径下文件夹数量
filelist = os.listdir("./faceImagesGray")
for item in filelist:    
    num_dirs +=1
    
#data = tf.placeholder(tf.float64, [None, 10000]) # 100x100
traindata=np.zeros((num_dirs*480,784))

trainlabel=np.zeros((num_dirs*480,num_dirs))
testdata=np.zeros((num_dirs*120,784))
testlabel=np.zeros((num_dirs*120,num_dirs))
h=0;j=0;j1=0
for i in filelist:
    filepath="D:\\faceImagesGray\\%s" %(i)
    Filelist = os.listdir(filepath)
    for r1 in range(0,120):
        testlabel[120*h+r1][h]=1
    for r in range(0,480):
        trainlabel[480*h+r][h]=1
    h=h+1;k1=1
    for k in Filelist:
        Filepath=filepath+""+"\\%s" %(k)
        img = Image.open(Filepath)
        pic = img.resize((28, 28))
        img_convert_ndarray = np.array(pic).reshape((1,784))
        if k1<=480:            
            traindata[j,:]=img_convert_ndarray
            j=j+1;k1=k1+1
        else:
            testdata[j1,:]=img_convert_ndarray
            j1=j1+1;k1=k1+1
            
            
#创建占位符
X=tf.placeholder(dtype=tf.float64,shape=[None,784])
y=tf.placeholder(dtype=tf.float64,shape=[None,num_dirs])

#卷积核，在卷积神经网络中，是变量
#变量生成方法
def gen_v(shape):
    return tf.Variable(initial_value=tf.random_normal(dtype=tf.float64,shape=shape,stddev=0.1),dtype=tf.float64)

#定义方法，完成卷积操作
def conv(input_data,filter_):
    return tf.nn.conv2d(input=input_data,filter=filter_,strides=[1, 1, 1, 1], padding='SAME')

#定义，池化操作
def pool(input_data):
    return tf.nn.max_pool(value=input_data,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#第一层卷积
input_data1=tf.reshape(X,shape=[-1,28,28,1])

#卷积核
filter1=gen_v(shape=[3,3,1,64])
conv1=conv(input_data1,filter1)

#偏差bias
b1=gen_v(shape=[64])
conv1=conv(input_data1,filter1)+b1

#池化
pool1=pool(conv1)

#激活函数
activel=tf.nn.relu(pool1)

#第二层卷积
#使用的是第一层卷积的数据
filter2=gen_v(shape=[3,3,64,64])
b2=gen_v(shape=[64])

conv2=conv(activel,filter2)+b2

#池化
pool2=pool(conv2)

#激活
active2=tf.nn.sigmoid(pool2)

#全连接层
#1024个方程，1024个神经元
fc_w=gen_v(shape=[7*7*64,1024])

fc_b=gen_v(shape=[1024])

conn=tf.matmul(tf.reshape(active2,shape=[-1,7*7*64]),fc_w)+fc_b

#conn=tf.matmul(tf.reshape(actice2,shape=[-1,7*7*64]),fc_2)+
#dropout防止过拟合
kp=tf.placeholder(dtype=tf.float64,shape=None)
dropout=tf.nn.dropout(conn,keep_prob=kp)


#输出层out
#10个类别0~9
out_w=gen_v(shape=[1024,num_dirs])
out_b=gen_v(shape=[num_dirs])

out=tf.matmul(dropout,out_w)+out_b

#概率，预测的概率，非真实分布
prob=tf.nn.softmax(out)

#真是概率是y


#交叉熵
cost=tf.reduce_mean(tf.reduce_sum(y*tf.log(1/prob),axis=-1))

#cost2=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits_v2(label=y,logits=prob))

adam=tf.train.AdamOptimizer()

optimizer=adam.minimize(cost)

#config=tf.ConfigProto()


saver=tf.train.Saver()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    
#for i in range(10):
    #c=0
    for j in range(100):
        X_train=traindata;y_train=trainlabel
        optimizer_,cost_=sess.run(fetches=[optimizer,cost],feed_dict={X:X_train,y:y_train,kp:0.5})
        #c+=cost_/100
        print('里层循环次数：%d,每次损失:%0.4f'%(j,cost_))
        
    #print('-----执行次数: %d,损失函数是:%0.4f----'%(i,c))
    saver.save(sess,save_path='./model/cnn/model',global_step=1)

with tf.Session() as sess:
    saver.restore(sess,'./model/cnn/model-1')
    X_test=testdata;y_test=testlabel
    prob_=sess.run(prob,feed_dict={X:X_test,kp:1.0})
    y_test=y_test.argmax(axis=-1)
    prob_=prob_.argmax(axis=-1)
    print((y_test==prob_).mean())
'''
#path="F:\\0.png" 
#img = Image.open(path)
#testdata1=np.zeros((1,784))
#pic = img.resize((28, 28))
#img_convert_ndarray = np.array(pic).reshape((1,784))
#testdata[0,:]=img_convert_ndarray
#with tf.Session() as sess:
#    saver.restore(sess,'./cnn/model-1')
#    X_test=testdata1
#    prob_=sess.run(prob,feed_dict={X:X_test,kp:1.0})     
'''