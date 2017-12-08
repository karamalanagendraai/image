import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests as r
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from PIL import Image, ImageTk
import csv
import caffe
from os import listdir
from os.path import isfile, join
import sys
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
le = joblib.load('filename_transformer.pkl')
Multi_class = joblib.load('filename_model.pkl')

caffe_root = 'caffe-windows/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
caffe.set_mode_cpu()

model_def = caffe_root + 'models/hybridCNN.tar/hybridCNN_deploy.prototxt'
model_weights = caffe_root + 'models/hybridCNN.tar/hybridCNN_iter_700000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

mu = np.load(caffe_root + 'models/hybridCNN.tar/hybridCNN_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

Dataset_df = pd.read_csv('test_gender.csv')

net.blobs['data'].reshape(Dataset_df.shape[0],        # batch size
                          3,         # 3-channel (BGR) images
                          227, 227)  # image size is 227x227

for i in range(0, Dataset_df.shape[0]):
    image = caffe.io.load_image(Dataset_df.iloc[i][0])
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[i,...] = transformed_image
    
net.forward()
Feature_df = pd.DataFrame(net.blobs['fc7'].data[...])

try:
    Y_predicted = Multi_class.predict(Feature_df)
    Y_test=Multi_class.predict_proba(Feature_df)

    i=0
    for item in Y_predicted:
        print ':'+le.inverse_transform(item), 'Score:'+str(Y_test[i,item])  
        i=i+1
except e:
    pass

