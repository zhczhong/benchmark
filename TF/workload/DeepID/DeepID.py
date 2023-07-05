#!/usr/bin/env python
# coding: utf-8

import time
import argparse

import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Conv2D, Activation, Input, Add
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.pooling import MaxPooling2D
from tensorflow.keras import mixed_precision

#to detect and align faces and find distances
#!pip install deepface
from deepface.commons import functions, distance as dst


parser = argparse.ArgumentParser(description='arguements')
parser.add_argument('--precision', type=str, default="float32", help='precision, float32, int8, bfloat16')
args = parser.parse_args()

# # DeepID 
# 
# This model is developed by The Chinese University of Hong Kong
# 
# Ref: http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf

if args.precision == "bfloat16":
    mixed_precision.set_global_policy('mixed_bfloat16')
    print("---- Use mixed bfloat16 precision: ")

def build_deepid_model():
    
    myInput = Input(shape=(55, 47, 3))
    
    x = Conv2D(20, (4, 4), name='Conv1', activation='relu', input_shape=(55, 47, 3))(myInput)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool1')(x)
    x = Dropout(rate=1, name='D1')(x)
    
    x = Conv2D(40, (3, 3), name='Conv2', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool2')(x)
    x = Dropout(rate=1, name='D2')(x)
    
    x = Conv2D(60, (3, 3), name='Conv3', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool3')(x)
    x = Dropout(rate=1, name='D3')(x)
    
    #--------------------------------------
    
    x1 = Flatten()(x)
    fc11 = Dense(160, name = 'fc11')(x1)
    
    #--------------------------------------
    
    x2 = Conv2D(80, (2, 2), name='Conv4', activation='relu')(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name = 'fc12')(x2)
    
    #--------------------------------------
    
    y = Add()([fc11, fc12])
    y = Activation('relu', name = 'deepid')(y)
    
    model = Model(inputs=[myInput], outputs=y)
    
    return model


model = build_deepid_model()

# # Loading pre-trained weights
# Even though the original study is shared the structure of the network publicly, pre-trained weights is not shared. Roy Ran in open source community re-trained the DeepID model and shared trained weights in TensorFlow format here: https://github.com/Ruoyiran/DeepID . This model got 99.39% accuracy for validation set and 97.05% accuracy for test set in the best epoch.
# 
# I convert TensorFlow weights to Keras. [Here](https://github.com/serengil/tensorflow-101/blob/master/python/deepid-tf-to-keras.ipynb), you can find how to convert TensorFlow weights to Keras.
# load pre-trained weights
model.load_weights("./deepid_keras_weights.h5")
model.summary()

target_size_x = model.layers[0].input_shape[0][2]
target_size_y = model.layers[0].input_shape[0][1]
print("model input shape is (",target_size_x," x ",target_size_y,")")


# # Data set
# Data set: https://github.com/serengil/deepface/tree/master/tests/dataset
idendities = {
    "Angelina": ["img2.jpg", "img4.jpg", "img6.jpg"],
    "Katy": ["img42.jpg", "img44.jpg", "img45.jpg"],
    "Scarlett": ["img9.jpg", "img48.jpg", "img49.jpg"],
}
positives = []
for key, values in idendities.items():
    
    #print(key)
    for i in range(0, len(values)-1):
        for j in range(i+1, len(values)):
            #print(values[i], " and ", values[j])
            positive = []
            positive.append(values[i])
            positive.append(values[j])
            positives.append(positive)

positives = pd.DataFrame(positives, columns = ["file_x", "file_y"])
positives["decision"] = "Yes"

samples_list = list(idendities.values())

negatives = []

for i in range(0, len(idendities) - 1):
    for j in range(i+1, len(idendities)):
        #print(samples_list[i], " vs ",samples_list[j]) 
        cross_product = itertools.product(samples_list[i], samples_list[j])
        cross_product = list(cross_product)
        #print(cross_product)
        
        for cross_sample in cross_product:
            #print(cross_sample[0], " vs ", cross_sample[1])
            negative = []
            negative.append(cross_sample[0])
            negative.append(cross_sample[1])
            negatives.append(negative)
            
negatives = pd.DataFrame(negatives, columns = ["file_x", "file_y"])
negatives["decision"] = "No"

negatives = negatives.sample(positives.shape[0], random_state=17)

df = pd.concat([positives, negatives]).reset_index(drop = True)
df.shape
df.decision.value_counts()
df.file_x = "./dataset/"+df.file_x
df.file_y = "./dataset/"+df.file_y
df.head()
df.shape

metrics = ["cosine", "euclidean", "euclidean_l2"]

for metric in metrics:
    df["DeepID_%s" % (metric)] = 0

total_time = 0.0
total_sample = 0
step = 0
#for index, instance in df.iterrows():
for index, instance in tqdm(df.iterrows(), total=df.shape[0]):
    img1_path = instance["file_x"]
    img2_path = instance["file_y"]
    
    # img1 = functions.detectFace(img1_path, (target_size_y, target_size_x))
    # img2 =  functions.detectFace(img2_path, (target_size_y, target_size_x))
    img1 = functions.preprocess_face(img1_path, (target_size_y, target_size_x))
    img2 = functions.preprocess_face(img2_path, (target_size_y, target_size_x))
    tic = time.time()
    img1_representation = model.predict(img1)[0,:]
    img2_representation = model.predict(img2)[0,:]
    toc = time.time()
    print("Iteration: {}, inference time: {} sec".format(step, toc - tic), flush=True)
    if step >= 5:
        total_time += (toc - tic)
        total_sample += 1
    
    step += 1
    for j in metrics:
        
        if j == 'cosine':
            distance = dst.findCosineDistance(img1_representation, img2_representation)
        elif j == 'euclidean':
            distance = dst.findEuclideanDistance(img1_representation, img2_representation)
        elif j == 'euclidean_l2':
            distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
        df.loc[index, 'DeepID_%s' % (j)] = distance

if total_time > 0 and total_sample > 0:
    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("Latency: {:.3f} ms".format(latency))
    print("Throughput: {:.2f} samples/s".format(throughput))


df

# for metric in metrics:
#     print(metric)
#     df[df.decision == "Yes"]['DeepID_%s' % (metric)].plot(kind='kde', title = metric, label = 'Yes', legend = True)
#     df[df.decision == "No"]['DeepID_%s' % (metric)].plot(kind='kde', title = metric, label = 'No', legend = True)
#     plt.show()

df.head()

def showInstance(idx):
    fig = plt.figure(figsize=(10,3))

    ax1 = fig.add_subplot(1,3,1)
    # plt.imshow(functions.detectFace(df.iloc[idx].file_x, (224, 224))[0][:,:,::-1])
    plt.imshow(functions.preprocess_face(df.iloc[idx].file_x, (224, 224))[0][:,:,::-1])
    plt.axis('off')

    ax2 = fig.add_subplot(1,3,2)
    # plt.imshow(functions.detectFace(df.iloc[idx].file_y, (224, 224))[0][:,:,::-1])
    plt.imshow(functions.preprocess_face(df.iloc[idx].file_y, (224, 224))[0][:,:,::-1])
    plt.axis('off')

    ax3 = fig.add_subplot(1,3,3)
    plt.text(0, 0.6, "Cosine: %s" % (round(df.iloc[idx].DeepID_cosine,4)))
    plt.text(0, 0.5, "Euclidean: %s" % (round(df.iloc[idx].DeepID_euclidean,4)))
    plt.text(0, 0.4, "Euclidean L2: %s" % (round(df.iloc[idx].DeepID_euclidean_l2,4)))
    plt.axis('off')

    plt.show()

# for i in df[df.decision == 'Yes'].sample(5, random_state=666).index.tolist():
#     showInstance(i)# 
# 

# # In[390]:# 
# 

# for i in df[df.decision == 'No'].sample(5, random_state=17).index.tolist():
#     showInstance(i)

print("Cosine: ")
threshold = 0.015

print("actual values of verified ones: ",df[df['DeepID_cosine'] <= threshold].decision.values)
print("actual values of unverified ones: ",df[df['DeepID_cosine'] > threshold].decision.values)

print("------------------------")

print("Euclidean:")
threshold = 45

print("actual values of verified ones: ",df[df['DeepID_euclidean'] <= threshold].decision.values)
print("actual values of unverified ones: ",df[df['DeepID_euclidean'] > threshold].decision.values)

print("------------------------")

print("Euclidean L2:")
threshold = 0.17

print("actual values of verified ones: ",df[df['DeepID_euclidean_l2'] <= threshold].decision.values)
print("actual values of unverified ones: ",df[df['DeepID_euclidean_l2'] > threshold].decision.values)

