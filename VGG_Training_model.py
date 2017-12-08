

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# display plots in this notebook
#%matplotlib inline

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap


#import sys
#caffe_root = 'caffe-windows/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
#sys.path.insert(0, caffe_root + 'python')

import caffe
import os

# set data root directory, e.g:
#data_root = osp.join('/Python Scripts/', 'Test_Database1')
#data_root = 'Current_Folder4/'
data_root = '/'

# these are the PASCAL classes, we'll need them later.
classes = np.array(['Bike Helmets', 'Child Carseat',  'Child Harness', 'Dangerous Chemicals', 'E-cigarettes Products',  'Hate Crime', 'Hoverboard', 'Lighters', 'Nudity','Plant Seeds', 'Recalled Toys','Tattoo Gun', 'Weapon'])


caffe.set_mode_cpu()

model_def = 'VGG_ILSVRC_16_layers_deploy.prototxt'
model_weights = 'VGG_ILSVRC_16_layers.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load(caffe_root + 'python/caffe/imagenet/hybridCNN_mean.npy')
mu = np.load('ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

from os import listdir
from os.path import isfile, join

Dataset_df = pd.DataFrame()

for cl in classes:
    img_dir = data_root+cl
    onlyfiles = [(img_dir+'/'+ f) for f in listdir(img_dir) if isfile(join(img_dir, f))]
    temp_df = pd.DataFrame(onlyfiles)
    temp_df['target']= cl
    Dataset_df = Dataset_df.append(temp_df,ignore_index=True)

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)

net.blobs['data'].reshape(1,#Dataset_df.shape[0],        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227
Feature_df = pd.DataFrame()                          
for i in range(0, Dataset_df.shape[0]):
    image = caffe.io.load_image(Dataset_df.iloc[i,0])
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[0,...] = transformed_image
    net.forward()
    temp_df = pd.DataFrame(net.blobs['fc7'].data[...])
    Feature_df = Feature_df.append(temp_df)
    print i
#net.forward()
#Feature_df1 = pd.DataFrame(net.blobs['fc7'].data[...])
#Feature_df = pd.DataFrame(net.blobs['fc7'].data[...])
#Feature_df = pd.concat((Feature_df1,Feature_df2),ignore_index=True,axis=1)
#Dataset_df = pd.concat((Dataset_df, Feature_df),ignore_index=True,axis=1)
Dataset_train_df = pd.DataFrame()
Dataset_test_df = pd.DataFrame()

#Dataset_train_df['X'], Dataset_test_df['X'], Dataset_train_df['target'], Dataset_test_df['target'] 
#X_train, X_test, Y_train, Y_test= train_test_split(Dataset_df.iloc[:,2:], Dataset_df.iloc[:,1], test_size=0.30, random_state=42)
X_train, X_test, Y_train, Y_test= train_test_split(Feature_df.iloc[:,2:], Dataset_df.iloc[:,1], test_size=0.30, random_state=42)

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(Dataset_df.iloc[:,1])
clf1 = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=10, min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features=0.5, max_leaf_nodes=None, bootstrap=True, oob_score=True, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight='balanced')
#Multi_class= OneVsRestClassifier(SVC(random_state=0))
#Multi_class= OneVsOneClassifier(SVC(random_state=0))

Multi_class= OneVsRestClassifier(clf1)
Multi_class.fit(X_train, le.transform(Y_train) )
Y_predicted = Multi_class.predict(X_test)

print np.sum(Y_predicted==le.transform(Y_test))
#confusion_matrix(Y_predicted,le.transform(Y_test))
from sklearn.externals import joblib
#joblib.dump(Multi_class, 'filename_model.pkl') 
#joblib.dump(le, 'filename_transformer.pkl') 
    
    
#Y_predicted = Multi_class.predict(X_test)
#joblib.dump(le, 'filename_transformer.pkl') 