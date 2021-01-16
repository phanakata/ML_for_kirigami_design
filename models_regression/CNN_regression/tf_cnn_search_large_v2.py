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
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

import random
import pickle
from time import time 

with open ('C25.11.14.wo_detached_shuffled.dat', 'rb') as fp:
    listBinary_all = pickle.load(fp)
    
#load list genome from data that has been simulated 
nsimulations = 10
list_listBinary=[]
data=[]
for j in range(0, nsimulations):
    with open ('list'+str(j)+'_cnn.dat', 'rb') as fp:
        listj = pickle.load(fp)
    data.append(load_and_tabulate_data_wo_s('ss-i_25_'+str(j)+'_cnn.dat', listj))
    list_listBinary +=listj

#print (data[0].shape)
#print (data[1].shape)
    
alldata_25G=np.concatenate((data[0], data[1]), axis=0)
for i in range (2, nsimulations):
    alldata_25G=np.concatenate((alldata_25G, data[i]) , axis=0)

print (len(data))
#alldata_25G = data[0]

#separate the unexplored data (this code is still inefficient)
nfeatures = len(listBinary_all[0]) #length of 2D arrays flatten to 1D
dataRem = np.zeros((len(listBinary_all)-len(alldata_25G), nfeatures)) # '+3' as we include 3 properties 
j=0     
for i in range(len(listBinary_all)):
    if listBinary_all[i] not in list_listBinary:
        dataRem[j][0:nfeatures] = toArray(listBinary_all[i])
        j= j + 1
    #data[i][nfeatures:] = rawData[i][1:]
    
print(len(listBinary_all), len(list_listBinary), len(dataRem))

#paramters to make finer grids 
NCcell_x = 5
NCcell_y = 5
ncell_x = 54
ncell_y = 80

#create fine grids 
listFG=[]
for i in range (len(alldata_25G)):
    cutConfigurations=alldata_25G[i, 0:-3]
    inner = generateInnerCell(NCcell_x, NCcell_y, ncell_x, ncell_y)
    inner_wCuts = makeCutsonCell(cutConfigurations, inner, NCcell_x, NCcell_y, ncell_x, ncell_y)
    listFG.append(inner_wCuts)

alldata_FG = np.array(listFG)
alldata_FG = np.append(alldata_FG, alldata_25G[:, -3:], 1)

alldata = alldata_FG #unflag this for fine grid

nfeatures = len(alldata[0])-3
nfeatures2 = len(alldata_25G[0])-3

#nfeautures is needed later to split the matrix
#print("Number of data:", len(alldata))
#print("Number of features (or inputs/grids):", nfeatures)


ntrain = len(alldata)
print ("Ntrain:", ntrain)
topAve, n_added_data, n_dropped_data=100, 100, 0

dataTrain= alldata_25G[:ntrain]
#dataRand= alldata_25G[:ntrain]


#add theree columns with zeros to dataRem as values are not found yet
unknown_values = np.zeros((len(dataRem), 3))
dataRem = np.append(dataRem, unknown_values, 1)
#NEED to shuffle them, in case for predicting not THE entire space
np.random.shuffle(dataRem)



#dataRem = alldata_25G[ntrain:]
#dataRandRem = alldata_25G[ntrain:]

listNN =[]
#listRand=[]


    
dataTrain = dataTrain[dataTrain[:,nfeatures2].argsort()] 
#get rid bottom 50 
fracture = dataTrain[int(len(dataTrain)//2), nfeatures2]
listNN.append([len(alldata)//100, np.mean(dataTrain[len(dataTrain)-topAve:, nfeatures2]), fracture])

#print (listRand)
print (listNN[0][0], listNN[0][1], listNN[0][2])

X_train = alldata[:ntrain, :nfeatures] 
y_train = alldata[:ntrain, nfeatures] 



y_train = np.reshape(y_train, (len(y_train), 1))



def CNN_regressor(hl, lr, bs, te, dr ):
    total_len = X_train.shape[0]

    # Parameters
    learning_rate = lr # 0.001
    training_epochs = te # 300 (max iteration)
    batch_size = bs #200 
    dropout_rate = dr #0.0
    display_step = 100000 


    # Network Parameters
    n_hidden_1 = hl[0] # 1st layer number of features
    #n_hidden_2 = 200 # 2nd layer number of features
    #n_hidden_3 = 200
    #n_hidden_4 = 256
    n_input = X_train.shape[1]
    n_classes = 1

    # tf Graph input

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 1])
    
    def conv_net(x):
        # Define a scope for reusing the variables
        with tf.name_scope('ConvNet'):
            x = tf.reshape(x, shape=[-1, 50, 80, 1])
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=x,
                filters=16,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)
        
            # Convolution Layer with 32 filters and a kernel size of 3
            #conv1 = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
            #print (conv1.shape)

            # Convolution Layer with 64 filters and a kernel size of 3
            conv2 = tf.layers.conv2d(conv1, 32, 3, activation=tf.nn.relu)
            
            #print(conv2.shape)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
            #print(conv2.shape)
            #conv3 = tf.layers.conv2d(conv2, 64, 2, activation=tf.nn.relu)
            
            conv3 = tf.layers.conv2d(conv2, 64, 3, activation=tf.nn.relu)
            
            #print(conv2.shape)
            # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
            conv3 = tf.layers.max_pooling2d(conv3, 2, 2)


            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv3)
            #print (fc1.shape)

            # Fully connected layer (in tf contrib folder for now)
            fc1 = tf.layers.dense(fc1, n_hidden_1, activation=tf.nn.relu)
            #layer_1 = tf.nn.relu(layer
            #n_hidden_1 =1024
            #layer_1 = fc1
            
            W_O = tf.Variable(tf.random_normal([n_hidden_1, n_classes], 0, 0.1)) 
            #tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1))
            #b_1 = tf.Variable(tf.zeros([32]))
            b_O = tf.Variable(tf.random_normal([n_classes], 0, 0.1))
        
            out = tf.add(tf.matmul(fc1,W_O), b_O)
        
    
        


            return out
  



    # Create model
    def model(x, weights, biases):
        with tf.name_scope("dnn"):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
            return out_layer

    # Store layers weight & bias
    with tf.name_scope("weights-biases"):
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1)),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], 0, 0.1))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1)),
            'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1))
        }

    # Construct model
    with tf.name_scope("train"):
        y_pred = conv_net(x)
        cost = tf.reduce_mean(tf.square(y_pred-y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Launch the graph
    #listRMSE =[]
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Training cycle
        total_batch = int(total_len/batch_size)
        #print ("num batch:", total_batch)
        for epoch in range(training_epochs):
            avg_cost = 0.
        
            # Loop over all batches
            for i in range(total_batch-1):
                batch_x = X_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train[i*batch_size:(i+1)*batch_size]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, p = sess.run([optimizer, cost, y_pred], feed_dict={x: batch_x,y:
                                                                         batch_y})
                # Compute average loss
                avg_cost += c / total_batch


        # Test model
        #print("Done training")
        listP = []
        #maxIte = len(dataRem)//100
        #maxIte = int(10000/100)
        array0 = np.zeros(len(dataRem))
        p_values = array0.reshape(len(array0),1) 
        maxIteration = len(dataRem)//100
        #maxIteration = 1000
        for j in range(maxIteration):
            listFG=[]
            start = j * 100
            end = (j+1) *100
            #print ("Batch:", j)
            for index in range (start, end):
                cutConfigurations=dataRem[index, 0:-3]
                inner = generateInnerCell(NCcell_x, NCcell_y, ncell_x, ncell_y)
                inner_wCuts = makeCutsonCell(cutConfigurations, inner, NCcell_x, NCcell_y, ncell_x, ncell_y)
                listFG.append(inner_wCuts)

            alldata_FGR = np.array(listFG)
            dataRem_expanded = np.append(alldata_FGR, dataRem[start:end, -3:], 1)
            #only keep to100
            X_test = dataRem_expanded[:, :nfeatures]
            y_test = dataRem_expanded[:, nfeatures]
            y_test = np.reshape(y_test, (len(y_test), 1))

            
            accuracy = sess.run(cost, feed_dict={x: X_test, y: y_test})
            predicted_vals = sess.run(y_pred, feed_dict={x: X_test})
            r2 = r2_score(y_test, predicted_vals)
            
        
            for kk in range(len(predicted_vals)):
                p_values [start+kk, 0] = predicted_vals[kk] 
            
            #list0 = list(predicted_vals)
            #listP = listP +list0
        #print(predicted_vals.shape)
        #print ("hl, r2, rmse:", n_hidden_1, r2_score(y_test, predicted_vals), np.sqrt(accuracy))
        return p_values
    
y_predicted_vals = CNN_regressor(hl=(64, ), lr=0.001, bs=100, te=200, dr=0.0)


dataRem = np.append(dataRem, y_predicted_vals, 1)
 
arr = dataRem[dataRem[:,nfeatures2+3].argsort()] #index 18 prob bad design, small -> goode design
        #choose top 5% 
nextData = arr[len(arr)-n_added_data:, :nfeatures2+3]

nextList =[]
for i in range (len(nextData)):
    nextList.append(toString(nextData[i, 0:nfeatures2]))
    
with open('list10_cnn.dat', 'wb') as fp:
    pickle.dump(nextList, fp)
    
#np.savetxt('junk.dat', np.array(nextData))    
