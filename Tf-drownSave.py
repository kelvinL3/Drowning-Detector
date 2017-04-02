import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as pylab
import cv2
import math

input_num_units=128*128
output_num_units=2
root_dir = os.path.abspath('./')
data_dir = root_dir

def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    #print(labels_dense)
    #print(num_labels)
    #print(num_classes)
    labels_one_hot = np.zeros((num_labels, num_classes))
    #print(labels_one_hot)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def checkBlock(img,hidden_layer,output_layer,y):
    temp=[]
    img = img.astype('float32')
    temp.append(img)
    print ("what")
    test_x = np.stack(temp)

    
    
        
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
##    val_x = test_x.item(0)
##    
##    booleanToReturn = (output_layer.eval(val_x) == y)
    pred_temp = tf.reduce_mean(tf.cast(pred_temp, "float"))
    bol = pred_temp.eval({x: test_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)})
    return math.floor(bol+.5)
    #copy this line
    #pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    #accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    #print(accuracy)




# 1280 x 720
# 1280/32 = 40
# 720/32 = 22.5 ~ 22
def checkFrame(img,hidden_layer,output_layer,y):

    for i in range(22):
        for j in range(40):

            box = img[i*32:i*32+128, j*32:j*32+128]

            drownFound = checkBlock(box,hidden_layer,output_layer,y)

            if(drownFound):
                cv2.rectangle(img, (j*32, i*32), (j*32+1, i*32+1), (0,255,0), 1)
                cv2.rectangle(img, (j*32+128, i*32), (j*32+1+128, i*32+1), (0,255,0), 1)
                cv2.rectangle(img, (j*32, i*32+128), (j*32+1, i*32+1+128), (0,255,0), 1)
                cv2.rectangle(img, (j*32+128, i*32+128), (j*32+1+128, i*32+1+128), (0,255,0), 1)


    return img


###########










x = tf.placeholder(tf.float32, [None, input_num_units])

#do we need y, placeholder?
y = tf.placeholder(tf.float32, [None, output_num_units])

val_y = np.array([1,0])





init = tf.global_variables_initializer()
with tf.Session() as sess:
##    # get training data
##    new_saver = tf.train.import_meta_graph(os.path.join(data_dir, 'DrowningImages', 'my-model.meta'))
##    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
##    #new_saver = tf.train.Saver()
##    #new_saver.restore(sess, os.path.join(data_dir, 'DrowningImages', 'my-model'))
##    print("a")
##    all_vars = tf.get_collection('vars')
##    
##    for v in all_vars:
##        print("b")
##        v_ = sess.run(v)
##        print(v)
    sess.run(init)
    print("START")
    hidWArray = np.loadtxt("weighthidden.txt").astype(np.float32)
    outWArray = np.loadtxt("weightout.txt").astype(np.float32)
    print("LOADED")
    hidWTensor = tf.Variable(hidWArray)
    outWTensor = tf.Variable(outWArray)
    print(hidWArray[0][0])
    print(outWArray[0][0])

    hidBArray = np.loadtxt("biaseshidden.txt").astype(np.float32)
    outBArray = np.loadtxt("biasesout.txt").astype(np.float32)

    hidBTensor = tf.Variable(hidBArray)
    outBTensor = tf.Variable(outBArray)
    
    print("1")
    hidden_layer = tf.add(tf.matmul(x, hidWTensor), hidBTensor)
    hidden_layer = tf.nn.relu(hidden_layer)
    print("2")
    output_layer = tf.matmul(hidden_layer, outWTensor) + outBTensor
    print("3")

    img = cv2.imread("image.jpg")
    cv2.imshow("before",img)
    img = checkFrame(img,hidden_layer,output_layer,y)
    cv2.imshow("out", img)
    cv2.waitKey(0)
    











