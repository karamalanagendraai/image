import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import caffe
import cv2
caffe.set_mode_cpu()

caffe_root = '/home/knagendra/CA/'#sys.path.insert(0, caffe_root + 'python')


MODEL_DEPLOY_FILE = 'deploy.prototxt'
MODEL_WEIGHT_FILE = 'GoogleNet_SOS.caffemodel'

net = caffe.Net(MODEL_DEPLOY_FILE,      # defines the structure of the model
                MODEL_WEIGHT_FILE,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
#mu = np.load('ResNet_mean.binaryproto')
#mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
#print 'mean-subtracted values:', zip('BGR', mu)

MODEL_ORIGINAL_INPUT_SIZE = 256, 256
MODEL_INPUT_SIZE = 224, 224
MODEL_MEAN_FILE = 'ResNet_mean.binaryproto'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open(MODEL_MEAN_FILE, 'rb').read()
blob.ParseFromString(data)
MODEL_MEAN_VALUE = np.squeeze(np.array( caffe.io.blobproto_to_array(blob) ))


net = caffe.Net(MODEL_DEPLOY_FILE,      # defines the structure of the model
                MODEL_WEIGHT_FILE,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
                
                
#mu = np.load('ResNet_mean.npy')
#mu = mu.mean(1).mean(1)
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', MODEL_MEAN_VALUE)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,#Dataset_df.shape[0],        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227
image = caffe.io.load_image('bird.jpg')
transformed_image = transformer.preprocess('data', image)
transformed_image=transformed_image[:,:,1]-104
transformed_image=transformed_image[:,:,2]-117
transformed_image=transformed_image[:,:,3]-123




net.blobs['data'].data[...] = transformed_image
output = net.forward()
output_prob = output['prob'][0] 
output_prob.argmax()

label_mapping = np.loadtxt("ILSVRC_2014.txt", str, delimiter='\t')



best_n = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
text=label_mapping[best_n[:]]
best_prob=output_prob[best_n[:]] 

loc=40
for a in range(len(text)):
    cv2.putText(image, "{}".format(best_prob[a]), (0, loc),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)    
    cv2.putText(image, "{}".format(text[a].split(' ')[1:]), (300, loc),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    loc=loc+40                       
        
cv2.imshow("Image", image)
key = cv2.waitKey(0)
