
import tensorflow as tf

from skimage import io,transform
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
#-----------------构建网络----------------------

x=tf.placeholder(tf.float32,shape=[None,100,100,3],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

#第一个卷积层（100——>50)
conv1=tf.layers.conv2d(
      inputs=x,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1=tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

#第二个卷积层(50->25)
conv2=tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2=tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

#第三个卷积层(25->12)
conv3=tf.layers.conv2d(
      inputs=pool2,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool3=tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

#第四个卷积层(12->6)
conv4=tf.layers.conv2d(
      inputs=pool3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool4=tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

#全连接层
dense1 = tf.layers.dense(inputs=re1,
                      units=1024,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
dense2= tf.layers.dense(inputs=dense1,
                      units=512,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
logits= tf.layers.dense(inputs=dense2,
                        units=5,
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.nn.l2_loss)

#---------------------------网络结束---------------------------
#取出所有参与训练的参数
params=tf.trainable_variables()
print("训练的参数为:")

#循环列出参数
for idx, v in enumerate(params):
     print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))


#读取数据  只能保存变量，不能保存神经网络的框架。保存权重有利于下一次的训练，或者可以用这个数据进行识别
#文件的位置
iris = pd.read_csv('data\d-1.csv', usecols=[1, 2, 3, 4, 5])
iris.info()


