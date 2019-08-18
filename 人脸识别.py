# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:54:05 2019

@author: Administrator
"""
import os
import tensorflow as tf
#import cv2
from PIL import Image
import numpy as np
#from mtcnn.mtcnn import MTCNN
#from tensorflow.examples.tutorials.mnist import input_data
#import tensorflow as tf
num_dirs=3
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
'''
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
    saver.save(sess,save_path='./cnn/model',global_step=1)

with tf.Session() as sess:
    saver.restore(sess,'./cnn/model-1')
    X_test=testdata;y_test=testlabel
    prob_=sess.run(prob,feed_dict={X:X_test,kp:1.0})
    y_test=y_test.argmax(axis=-1)
    prob_=prob_.argmax(axis=-1)
    print((y_test==prob_).mean())
'''
filelist = os.listdir("./faceImagesGray")
k1=0
path="F:\\rlsb.png" 
img = Image.open(path)
testdata1=np.zeros((1,784))
pic = img.resize((28, 28))
img_convert_ndarray = np.array(pic).reshape((1,784))
testdata1[0,:]=img_convert_ndarray
with tf.Session() as sess:
    saver.restore(sess,'./cnn/model-1')
    X_test=testdata1
    prob_=sess.run(prob,feed_dict={X:X_test,kp:1.0})    
    prob_=prob_.argmax(axis=-1)
    for item in filelist:
        if k1==prob_:
            jg=item
            break
        else:
            k1=k1+1
