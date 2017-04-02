import os
import numpy as np
import pandas as pd
from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
import matplotlib.pyplot as pylab
import json

# To stop potential randomness
seed = 130
rng = np.random.RandomState(seed)

root_dir = os.path.abspath('./')
data_dir = root_dir
# //sub_dir = os.path.join(root_dir, 'sub')

# check for existence
os.path.exists(root_dir)
os.path.exists(data_dir)
# //os.path.exists(sub_dir)

print ('1')

train = pd.read_csv(os.path.join(data_dir, 'output.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))


train.head()

print ('2')

img_name = rng.choice(train.filename)
filepath = os.path.join(data_dir, 'Images', img_name)

img = imread(filepath, flatten=False)

print ('3')

print ('4')
fileNum = 0
temp = []


#copy this, but not really
for img_name in train.filename:
    print(str(fileNum/50000))
    image_path = os.path.join(data_dir, 'Images', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    fileNum += 1
    
train_x = np.stack(temp)
#



temp = []
for img_name in test.filename:
    image_path = os.path.join(data_dir, 'Images', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)
    
test_x = np.stack(temp)

split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train.label.values[:split_size], train.label.values[split_size:]

#tf.add_to_collection('vars', train_x)
#tf.add_to_collection('vars', train_y)
#tf.add_to_collection('vars', tf.Variable(val_x))
#tf.add_to_collection('vars', tf.Variable(val_y))
print((val_y))





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

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    
    batch_y = eval(dataset_name).ix[batch_mask, 'label'].values
    batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y

input_num_units = 128*128
hidden_num_units = 5000
output_num_units = 2

x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

tf.add_to_collection('vars', x)
tf.add_to_collection('vars', y)

print ('5')

epochs = 2
batch_size = 128
learning_rate = 0.015


weights = {
    'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units], seed=seed))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

#tf.add_to_collection('vars', weights['hidden'])
#tf.add_to_collection('vars', weights['output'])
#tf.add_to_collection('vars', biases['hidden'])
#tf.add_to_collection('vars', biases['output'])

print ('6')


#copy these, weights['hidden'] with hidtensor
#todo save an load biases in same way as weights
hidden_layer = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
hidden_layer = tf.nn.relu(hidden_layer)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']



tf.add_to_collection('vars', hidden_layer)
tf.add_to_collection('vars', output_layer)

print ('7')

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

#tf.add_to_collection('vars', cost)

print(learning_rate)
print(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

print('7.5')

##initialize the numbers to random stuff
init = tf.global_variables_initializer()

print ('8')

with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize



    ##training
    for epoch in range(epochs):
        avg_cost = 0
        total_batch = int(train.shape[0]/batch_size)
        print ('b')
        for i in range(total_batch):
            print ('c')
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train')
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            print(c)            
            avg_cost += c / total_batch
            
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    
    print ("\nTraining complete!")
    #saving training data
    #saver_a = tf.train.Saver()
    #print(saver_a.save(sess,os.path.join(root_dir, 'my-model')))

    



    #below block is copied into drownSave
    # find predictions on val set
    #testing\\


    #copy this line
    pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
    print(accuracy)

    #eval this line, replace val_x with pred_temp
    print ("Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))


    
    predict = tf.argmax(output_layer, 1)
    pred = predict.eval({x: test_x.reshape(-1, input_num_units)})

    out = weights['output'].eval()
    hid = weights['hidden'].eval()

    biashid = biases['hidden'].eval()
    biasout = biases['output'].eval()





np.savetxt("weighthidden.txt", hid, fmt = "%1.3f")
np.savetxt("weightout.txt", out, fmt = "%1.3f")

np.savetxt("biaseshidden.txt", biashid, fmt = "%1.3f")
np.savetxt("biasesout.txt", biasout, fmt = "%1.3f")



