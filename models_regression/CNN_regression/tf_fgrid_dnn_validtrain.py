"""
TensorFlow python code
CNN for regression 
"""
from __future__ import print_function

#import tensorflow
import tensorflow as tf

import sys 
sys.path.append("../../tools/")
#module to handle mapping strings <-> arrays
from helper_functions import * 
#module to read and preprocess raw data
from preprocess import * 
from generate_lattice import * 
#from tf_regressor import * 

#standard module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

#load the 15grid data 
alldata_15G=np.loadtxt('../../mddata/15grid_shuffled.dat')

#paramters to make finer grids 
NCcell_x = 3
NCcell_y = 5
ncell_x = 34
ncell_y = 80

#create fine grids 
listFG=[]
for i in range (len(alldata_15G)):
    cutConfigurations=alldata_15G[i, 0:-3]
    inner = generateInnerCell(NCcell_x, NCcell_y, ncell_x, ncell_y)
    inner_wCuts = makeCutsonCell(cutConfigurations, inner, NCcell_x, NCcell_y, ncell_x, ncell_y)
    listFG.append(inner_wCuts)

alldata_FG = np.array(listFG)
alldata_FG = np.append(alldata_FG, alldata_15G[:, -3:], 1)
#the last three columns are yield strain, toughness, and yield stress. 

#alldata = alldata_15G
alldata = alldata_FG #unflag this for fine grid

nfeatures = len(alldata[0])-3 #nfeautures is needed later to split the matrix
print("Number of data:", len(alldata))
print("Number of features (or inputs/grids):", nfeatures)



#DON'T suffle. Data is already pre-shuffled to make sure same test set is used for each model 
x, y=create_matrix(alldata, False, 0, 0.375, nfeatures)
X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(x, y, 0.8, 0.1)
print (len(y_valid), len(y_test))


y_train = np.reshape(y_train, (len(y_train), 1))
y_valid = np.reshape(y_valid, (len(y_valid), 1))
y_test = np.reshape(y_test, (len(y_test), 1))

def CNN_regressor(hl, lr, bs, te, dr ):
    total_len = X_train.shape[0]

    # Parameters
    learning_rate = lr 
    training_epochs = te 
    batch_size = bs  
    dropout_rate = dr 
    display_step = 100000 


    # Network Parameters
    n_hidden_1 = hl[0] # 1st layer number of features
    n_input = X_train.shape[1]
    n_classes = 1

    # tf Graph input

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 1])
    
    def conv_net(x):
        # Define a scope for reusing the variables
        with tf.name_scope('ConvNet'):
            x = tf.reshape(x, shape=[-1,30, 80, 1])
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=x,
                filters=16,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        
            # Conv. Layer with 16 filters and a kernel size of 3
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            #conv1 = tf.layers.average_pooling2d(x, 2, 2)

            # Conv. Layer with 32 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 32, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            
            # Conv. Layer with 64 filters and a kernel size of 3
            conv3 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2)

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv3)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, n_hidden_1, activation=tf.nn.relu)
            
            W_O = tf.Variable(tf.random_normal([n_hidden_1, n_classes], 0, 0.1)) 
            b_O = tf.Variable(tf.random_normal([n_classes], 0, 0.1))
        
            out = tf.add(tf.matmul(fc1,W_O), b_O)
        
    
        

            return out
  



    # Construct model and lauch graph
    with tf.name_scope("train"):
        y_pred = conv_net(x)
        cost = tf.reduce_mean(tf.square(y_pred-y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #listRMSE =[]
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Training cycle
        total_batch = int(total_len/batch_size)
        print ("num batch:", total_batch)
        for epoch in range(training_epochs):
            avg_cost = 0.
        
            # Loop over all batches
            for i in range(total_batch-1):
                batch_x = X_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train[i*batch_size:(i+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, y_pred], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch

            # sample prediction
            label_value = batch_y
            estimate = p
            err = label_value-estimate
        
            #listRMSE.append([epoch, np.sqrt(avg_cost)])
            # Display logs per epoch step
            if epoch % display_step == 0:
                print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                print ("[*]----------------------------")
                for i in range(10):
                    print ("label value:", label_value[i], "estimated value:", estimate[i])
                print ("[*]============================")
        # on train 
        accuracy_t = sess.run(cost, feed_dict={x: X_train, y: y_train})
        predicted_vals_t = sess.run(y_pred, feed_dict={x: X_train})
        r2_t = r2_score(y_train, predicted_vals_t)        
        # on validation 
        accuracy_v = sess.run(cost, feed_dict={x: X_valid, y: y_valid})
        predicted_vals_v = sess.run(y_pred, feed_dict={x: X_valid})
        r2_v = r2_score(y_valid, predicted_vals_v)
        # on test 
        accuracy = sess.run(cost, feed_dict={x: X_test, y: y_test})
        predicted_vals = sess.run(y_pred, feed_dict={x: X_test})
        r2 = r2_score(y_test, predicted_vals)
        print ("hl, r2, rmse:", n_hidden_1, r2_score(y_test, predicted_vals), np.sqrt(accuracy))

        
        return n_hidden_1, r2, np.sqrt(accuracy), r2_v, np.sqrt(accuracy_v), r2_t, np.sqrt(accuracy_t)


#make list of different architectures
list_hls = [(4, ), (8, ), (16, ), (32, ), (64, ), (128, ), (256, ), (512, ), (1024, ), (2048, ) ]

listRMSE = []
listRMSE_v = []
listRMSE_t = []


for i in range (len(list_hls)):
    nhidden, r2, rmse, r2_v, rmse_v, r2_t, rmse_t = NN_regressor(hl=list_hls[i], lr=0.0001, bs=200, te=300, dr=0.0)
    listRMSE.append([nhidden, r2, rmse])
    listRMSE_v.append([nhidden, r2_v, rmse_v])
    listRMSE_t.append([nhidden, r2_t, rmse_t])




np.savetxt('test_CNN_2400G_lr0.0001e300_f16k3m_f32k3m_f64k3m.dat2', np.array(listRMSE))
np.savetxt('valid_CNN_2400G_lr0.0001e300_f16k3m_f32k3m_f64k3m.dat2', np.array(listRMSE_v))
np.savetxt('train_CNN_2400G_lr0.0001e300_f16k3m_f32k3m_f64k3m.dat2', np.array(listRMSE_t))


print ("DONE")
