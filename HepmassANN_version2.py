# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:42:27 2017

@author: kashu
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
#lablesTrain=np.reshape(lablesTrain,7000000)
completeDataTrain = completeDataTrain[:,1:29]
lablesTest = completeDataTest[:,0:1]
#lablesTest=np.reshape(lablesTest,3500000)
completeDataTest = completeDataTest[:,1:29]
num_labels = 1
num_features =28
#lablesTrain = (np.arange(num_labels) == lablesTrain[:,None]).astype(np.float32)
#lablesTest = (np.arange(num_labels) == lablesTest[:,None]).astype(np.float32)
def accuracy(predictions, labels):
  return (100.0 * np.sum(predictions == labels)
          / predictions.shape[0])

train_subset = 10000

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.constant(completeDataTrain[:train_subset,:])
    tf_train_labels = tf.constant(lablesTrain[:train_subset])
    tf_valid_dataset = tf.constant(completeDataTrain[train_subset:20000,:])
    tf_test_dataset = tf.constant(completeDataTest[:train_subset,:])
    valid_labels = lablesTrain[train_subset:20000]
    test_labels = lablesTest[:train_subset]
    weights = tf.Variable(tf.truncated_normal([num_features,num_labels],mean=0.0,stddev=1.0,dtype=tf.float32))
    biases = tf.Variable(tf.zeros([num_labels]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.sigmoid(logits)
    valid_prediction = tf.nn.sigmoid(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights) + biases)
    
num_steps = 10001

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, lablesTrain[:train_subset, :]))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
            
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
    
    
twoThird = int((2/3)*completeDataTrain.shape[0])
trainingData = completeDataTrain[:twoThird]
trainingLabels = lablesTrain[:twoThird]
validData = completeDataTrain[twoThird:]
validLabels = lablesTrain[twoThird:]      
    
del dest_filenameTest,dest_filenameTrain,filenameTest,filenameTrain
del dataTest,dataTrain,data_root,lablesTrain,completeDataTrain

batch_size = 128

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(validData)
    tf_test_dataset = tf.constant(completeDataTest)
    weights = tf.Variable(tf.truncated_normal([num_features, num_labels],mean=0.0,stddev=1.0,dtype=tf.float32))
    biases = tf.Variable(tf.zeros([num_labels]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    train_prediction = tf.nn.sigmoid(logits)
    valid_prediction = tf.nn.sigmoid(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights) + biases)
    
    
num_steps = 5500001

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
    if (step % 500 == 0):
      print('offset',offset)  
      print(trainingLabels.shape[0])
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validLabels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), lablesTest))
  
#singe layer NN with Adam Optimizer  


batch_size = 128
#comments are with wrong numbers ,used just for reference
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(validData)
    tf_test_dataset = tf.constant(completeDataTest)
    weights = tf.Variable(tf.truncated_normal([num_features, num_labels],mean=0.0,stddev=1.0,dtype=tf.float32))
    biases = tf.Variable(tf.zeros([num_labels]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)
    train_prediction = tf.nn.sigmoid(logits)
    valid_prediction = tf.nn.sigmoid(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights) + biases)
    
    
num_steps = 2000001

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
    if (step % 500 == 0):
      print('offset',offset)  
      print(trainingLabels.shape[0])
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validLabels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), lablesTest))
  
   


twoThird = int((2/3)*completeDataTrain.shape[0])
trainingData = completeDataTrain[:5000]
trainingLabels = lablesTrain[:5000]
validData = completeDataTrain[twoThird:twoThird+1000]
validLabels = lablesTrain[twoThird:twoThird+1000]
testData = completeDataTest[:1000]
testLabels = lablesTest[:1000]

del dest_filenameTest,dest_filenameTrain,filenameTest,filenameTrain
del dataTest,dataTrain,data_root,lablesTrain,completeDataTrain,completeDataTest,lablesTest


num_nodes= 3
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
    weights_1 = tf.Variable(tf.zeros([num_features, num_nodes]))
    biases_1 = tf.Variable(tf.zeros([num_nodes]))
    weights_2 = tf.Variable(tf.zeros([num_nodes, num_labels]))
    biases_2 = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits_1 = tf.matmul(tf_train_dataset, weights_1) + biases_1
    
    #(128,784)*(784,1024)+(1024,1)= Dimension of logit_1 =(128,1)
    sigmoid_layer= tf.nn.sigmoid(logits_1)
    #output of sigmoid layer would be (128,1024)    
    logits_2 = tf.matmul(sigmoid_layer, weights_2) + biases_2
    #(128,1024)*(1024,10)+(10,1)= Dimension of logit_2 = (128,1)
    loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits_2))

    # Optimizer.
    optimizer = tf.train.AdadeltaOptimizer(0.05).minimize(loss)
#    global_step = tf.Variable(0)  # count the number of steps taken.
#    start_learning_rate = 0.5
#    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
#    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training
    train_prediction = tf.nn.sigmoid(logits_2)
    
    # Predictions for validation 
    logits_1 = tf.matmul(tf_valid_dataset, weights_1) + biases_1
    sigmoid_layer= tf.nn.sigmoid(logits_1)
    logits_2 = tf.matmul(sigmoid_layer, weights_2) + biases_2
    
    valid_prediction = tf.nn.sigmoid(logits_2)
    
    
    # Predictions for test
    logits_1 = tf.matmul(tf_test_dataset, weights_1) + biases_1
    sigmoid_layer= tf.nn.sigmoid(logits_1)
    logits_2 = tf.matmul(sigmoid_layer, weights_2) + biases_2
    
    test_prediction =  tf.nn.sigmoid(logits_2)
   
  
num_steps = 300000

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
      print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), validLabels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), testLabels))
  
  
  
        