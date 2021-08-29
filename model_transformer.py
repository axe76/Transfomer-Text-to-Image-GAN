# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:06:20 2021

@author: sense
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img

from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy import asarray
import numpy as np

import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.initializers import RandomNormal
from numpy.random import random
from tensorflow.keras import layers
from tensorflow.keras import Model
from numpy.random import randn
from numpy.random import randint
import time
from tensorflow.keras.layers import PReLU
from tensorflow.keras.utils import plot_model

from transformer_layer import Encoder

# Discriminator model

class Discriminator(tf.keras.Model):
    def __init__(self, caption_vocab):
        super(Discriminator, self).__init__()
        self.encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=len(caption_vocab),
                               maximum_position_encoding=10000)
        
        self.batch_norm = layers.BatchNormalization(momentum=0.5)
        self.leakyRelu = layers.LeakyReLU(0.2)
        self.gaussian_noise = layers.GaussianNoise(0.2)
        
        self.n_nodes = 3 * 64 * 64
        self.dense_in = layers.Dense(self.n_nodes)
        
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")
        self.conv3 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")#128
        self.conv4 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")#128
        self.conv5 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")#256
        self.conv6 = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same")#256
        self.conv7 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")#512
        
        self.flatten = layers.Flatten()
        self.dense_out = layers.Dense(1024)
        self.final_dense = layers.Dense(1)
        
    
    def call(self,img,cap,training,mask):
        batch_size = img.shape[0]
        in_label = self.encoder(cap,training,mask)
        
        #print('shape of in_label',in_label.shape)
    
        li = self.dense_in(in_label)#layers.Dense(n_nodes)(in_label)
        #print('shape of li',li.shape)
        li = tf.reshape(li, (batch_size,64,64,3))
        #li = layers.Reshape((64, 64, 3))(li)
    
        dis_input = img#layers.Input(shape=(64, 64, 3))
        #print('dis_input shape:',dis_input.shape)
    
        merge = tf.concat([dis_input, li],axis=-1) 
        #print("Merge shape:",merge.shape)
        discriminator = self.conv1(merge)
        discriminator = self.leakyRelu(discriminator)
        discriminator = self.gaussian_noise(discriminator)
        
        discriminator = self.conv2(discriminator)
        #print("Discrim shape",discriminator.shape)
        discriminator = self.batch_norm(discriminator)
        discriminator = self.leakyRelu(discriminator)
    
        discriminator = self.conv3(discriminator)
        #print("Discrim shape 2",discriminator.shape)
        discriminator = self.batch_norm(discriminator)
        discriminator = self.leakyRelu(discriminator)
    
        discriminator = self.conv4(discriminator)
        discriminator = self.batch_norm(discriminator)
        discriminator = self.leakyRelu(discriminator)
    
        discriminator = self.conv5(discriminator)
        discriminator = self.batch_norm(discriminator)
        discriminator = self.leakyRelu(discriminator)
    
        discriminator = self.conv6(discriminator)
        discriminator = self.batch_norm(discriminator)
        discriminator = self.leakyRelu(discriminator)
    
        discriminator = self.conv7(discriminator)
        discriminator = self.batch_norm(discriminator)
        discriminator = self.leakyRelu(discriminator)
    
        discriminator = self.flatten(discriminator)
    
        discriminator = self.dense_out(discriminator)
    
        discriminator = self.leakyRelu(discriminator)
    
        discriminator = self.final_dense(discriminator)
        
        return discriminator
    
    


class ResnetLayer(tf.keras.layers.Layer):
    def __init__(self,kernel_size,filters,strides):
        super(ResnetLayer, self).__init__()
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.batch_norm = layers.BatchNormalization(momentum=0.5)
        self.prelu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        self.add = layers.Add()
    
    def call(self,model):
        gen = model
        model = self.conv(model)
        model = self.batch_norm(model)
        model = self.prelu(model)
        model = self.conv(model)
        model = self.batch_norm(model)
        model = self.add([gen,model])
        return model
        

class Generator(tf.keras.Model):
    def __init__(self,caption_vocab):
        super(Generator, self).__init__()
        self.kernel_init = tf.random_normal_initializer(stddev=0.02)
        
        self.encoder = Encoder(num_layers=2, d_model=512, num_heads=8, dff=2048, input_vocab_size=len(caption_vocab),
                               maximum_position_encoding=10000)
        
        self.resnet = ResnetLayer(3, 64, 1)
        
        self.dense1 = layers.Dense(8192)
        self.n_nodes = 128 * 8 * 8
        self.dense2 = layers.Dense(self.n_nodes)
        
        self.conv1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.conv3 = layers.Conv2D(3, (3, 3), padding='same', activation='tanh')
        
        self.prelu = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        self.batch_norm = layers.BatchNormalization(momentum=0.5)
        
        self.add = layers.Add()
        
        self.conv_tr1 = layers.Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=self.kernel_init)
        self.conv_tr2 = layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=self.kernel_init)
        self.conv_tr3 = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=self.kernel_init)
        self.conv_tr4 = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=self.kernel_init)
        
        self.leakyRelu = layers.LeakyReLU(0.2)
        
        
    
    def call(self,random_input,text_input1,training,mask):
        
        text_input1 = self.encoder(text_input1,training,mask)
        
        batch_size = text_input1.shape[0]
        #print('batch_size: ',batch_size)
        
        text_layer1 = self.dense1(text_input1)
        text_layer1 = tf.reshape(text_layer1,(batch_size,8,8,128))
        #print('text1 shape after 1st dense:', text_layer1.shape)
        gen_input_dense = self.dense2(random_input)
        #print('gen input shape:',gen_input_dense.shape)
        generator = tf.reshape(gen_input_dense,(batch_size,8,8,128)) 
    
        merge =  tf.concat([generator, text_layer1], axis=-1) 
    
        model = self.conv1(merge)
        model = self.prelu(model)
    
        gen_model = model
    
        for _ in range(4):
          model = self.resnet(model)
    
        model = self.conv2(model)
        model = self.batch_norm(model)
        model = self.add([gen_model, model])
    
        model = self.conv_tr1(model)
        model = self.leakyRelu(model)
    
        model = self.conv_tr2(model)
        model = self.leakyRelu(model)
    
        model = self.conv_tr3(model)
        model = self.leakyRelu(model)
    
        model = self.conv_tr4(model)
        model = self.leakyRelu(model)
    
        model = self.conv3(model)
    
        return model
        



