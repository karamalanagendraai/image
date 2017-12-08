import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import caffe
import os
from sklearn.externals import joblib

classes = np.array(['X', 'Y'])

caffe.set_mode_cpu()

#model_def = 'hybridCNN_deploy.prototxt'
#model_weights = 'hybridCNN_iter_700000.caffemodel'

model_def = 'VGG_ILSVRC_16_layers_deploy.prototxt'
model_weights = 'VGG_ILSVRC_16_layers.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
#mu = np.load('hybridCNN_mean.npy')
mu = np.load('ilsvrc_2012_mean.npy')

mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

model=[]
for j in range(0, len(classes)):
     temp1=joblib.load(str('VGG/Model_'+str(j)+'.pkl'))
     model.append(temp1)
                        
# testing of data images

#data_test_root=''
#Dataset_ts = pd.DataFrame()
#img_dir = data_test_root
#onlyfiles = [(f) for f in os.listdir(img_dir)]
#temp_df = pd.DataFrame(onlyfiles)
#Dataset_df =temp_df;
#Dataset_df = pd.read_csv("URL.csv")

net.blobs['data'].reshape(1,#Dataset_df.shape[0],        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227
Feature_df = pd.DataFrame() 
                          
                          
for i in range(0, 0):
    #image = caffe.io.load_image(Dataset_df.iloc[i][0])
    image = caffe.io.load_image(Dataset_df.iloc[i,0]) 
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[0,...] = transformed_image
    net.forward()
    temp_df = pd.DataFrame(net.blobs['fc7'].data[...])
    Feature_df = Feature_df.append(temp_df)
   
result_ts=[] 
predicteds=[]
flag=0       
#with open("OUTPUT.csv", 'w') as csvfile:
fieldnames = ['Product', 'Score','idImage','ProductId','Description']
#    writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=fieldnames)
for m in range(0, len(Feature_df)):
    feature=pd.DataFrame()
    temp=Feature_df.iloc[m,:]
    feature=feature.append(temp)
    flag=0
    for j in range(0, len(classes)):
        tempM=model[j] 
        predicted=tempM.predict(feature)
        scr=tempM.predict_proba(feature)
        #predicteds.append(predictedscore)
        if(predicted and scr[0,1]>=0.75):
            scr=tempM.predict_proba(feature)
            result_ts.append(classes[j])
            flag=1
            break
        if (flag==1):
            print({'Product':classes[j], 'Score':scr[0,1],'idImage':str(Dataset_df.iloc[m][0]), 'ProductId':str(Dataset_df.iloc[m][1]), 'Description':str(Dataset_df.iloc[m][2])})
            #writer.writerow({'Product':classes[j], 'Score':scr[0,1],'idImage':str(Dataset_df.iloc[m][0]), 'ProductId':str(Dataset_df.iloc[m][1]), 'Description':str(Dataset_df.iloc[m][2])})





                
            