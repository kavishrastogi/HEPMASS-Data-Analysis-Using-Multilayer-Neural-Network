# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:58:17 2017

@author: kashu
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
filenameTrain = 'all_train.csv'
filenameTest = 'all_test.csv'
data_root = 'C:/MASTERS/AdvancedML/Exercise/all_massData'
dest_filenameTrain = os.path.join(data_root,filenameTrain)
dest_filenameTest = os.path.join(data_root,filenameTest)
dataTrain = pd.read_csv(dest_filenameTrain)
dataTest = pd.read_csv(dest_filenameTest)
completeDataTrain= np.array(dataTrain,dtype ='float32')
completeDataTest = np.array(dataTest,dtype ='float32')
lablesTrain = completeDataTrain[:,0:1]
lablesTrain=np.reshape(lablesTrain,7000000)
completeDataTrain = completeDataTrain[:,1:29]
lablesTest = completeDataTest[:,0:1]
lablesTest=np.reshape(lablesTest,3500000)
completeDataTest = completeDataTest[:,1:29]
num_labels = 2
num_features =28
lablesTrain = (np.arange(num_labels) == lablesTrain[:,None]).astype(np.float32)
lablesTest = (np.arange(num_labels) == lablesTest[:,None]).astype(np.float32)
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
          / predictions.shape[0])
  
#twoThird = int((2/3)*completeDataTrain.shape[0])
#trainingData = completeDataTrain[:twoThird]
#trainingLabels = lablesTrain[:twoThird]
#validData = completeDataTrain[twoThird:]
#validLabels = lablesTrain[twoThird:] 

twoThird = int((2/3)*completeDataTrain.shape[0])
trainingData = completeDataTrain[:twoThird]
trainingLabels = lablesTrain[:twoThird]
validData = completeDataTrain[twoThird:]
validLabels = lablesTrain[twoThird:]
testData = completeDataTest
testLabels = lablesTest

num_nodes_1= 10
num_nodes_2 = 3
batch_size = 128

graph = tf.Graph()
with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(validData)
    tf_test_dataset = tf.constant(testData)

    # Variables.
    weights_1 = tf.Variable(tf.zeros([num_features, num_nodes_1]))
    biases_1 = tf.Variable(tf.zeros([num_nodes_1]))
    weights_2 = tf.Variable(tf.zeros([num_nodes_1, num_nodes_2]))
    biases_2 = tf.Variable(tf.zeros([num_nodes_2]))
    
    #adding extra layer
    weights_3 = tf.Variable(tf.zeros([num_nodes_2, num_labels]))
    biases_3 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    #(128,784)*(784,1024)+(1024,1)= Dimension of logit_1 =(128,1)
    sigmoid_layer_1= tf.nn.sigmoid(logits_1)
    #output of sigmoid layer would be (128,1024)    
    logits_2 = tf.matmul(sigmoid_layer_1, weights_2) + biases_2
    #(128,1024)*(1024,10)+(10,1)= Dimension of logit_2 = (128,1)
    
    #adding extra layer
    
    sigmoid_layer_2= tf.nn.sigmoid(logits_2)
    
    logits_3 = tf.matmul(sigmoid_layer_2, weights_3) + biases_3
    
#    loss = tf.reduce_mean(
#            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))
    loss = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_3))

    # Optimizer.
    optimizer = tf.train.AdadeltaOptimizer(0.2).minimize(loss)
#    global_step = tf.Variable(0)  # count the number of steps taken.
#    start_learning_rate = 0.5
#    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training
    #train_prediction = tf.nn.sigmoid(logits_2)
    train_prediction = tf.nn.sigmoid(logits_3)
    
    # Predictions for validation 
    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    sigmoid_layer_1= tf.nn.sigmoid(logits_1)
    logits_2 = tf.matmul(sigmoid_layer_1, weights_2) + biases_2
    sigmoid_layer_2= tf.nn.sigmoid(logits_2)
    logits_3 = tf.matmul(sigmoid_layer_2, weights_3) + biases_3
    
    
    
    valid_prediction = tf.nn.sigmoid(logits_3)
    
    
    # Predictions for test
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    sigmoid_layer_1= tf.nn.sigmoid(logits_1)
    logits_2 = tf.matmul(sigmoid_layer_1, weights_2) + biases_2
    sigmoid_layer_2= tf.nn.sigmoid(logits_2)
    logits_3 = tf.matmul(sigmoid_layer_2, weights_3) + biases_3
    
    test_prediction =  tf.nn.sigmoid(logits_3)
    
    
num_steps = 100000

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (trainingLabels.shape[0] - batch_size)
    #offset is a kind of window whnever it is greater than 200000 it is taking reminder 
    #of (step(chnaging every iteration) * batch_size) % (train_labels.shape[0](fixedvalue=200000) - batch_size(128))
    # Generate a minibatch.
    batch_data = trainingData[offset:(offset + batch_size), :]
    batch_labels = trainingLabels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print('offset',offset)  
      print(trainingLabels.shape[0])
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      #plt.plot(accuracy(predictions, batch_labels), label = 'accuracy')
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validLabels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), testLabels))
  #plt.show()