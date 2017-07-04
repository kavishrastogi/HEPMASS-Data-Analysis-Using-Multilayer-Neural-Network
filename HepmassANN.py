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
lablesTrain=np.reshape(lablesTrain,7000000)
completeDataTrain = completeDataTrain[:,1:29]
lablesTest = completeDataTest[:,0:1]
lablesTest=np.reshape(lablesTest,3500000)
completeDataTest = completeDataTest[:,1:29]
num_labels = 2
lablesTrain = (np.arange(num_labels) == lablesTrain[:,None]).astype(np.float32)
lablesTest = (np.arange(num_labels) == lablesTest[:,None]).astype(np.float32)


train_subset = 15000

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.constant(completeDataTrain[:train_subset,:])
    tf_train_labels = tf.constant(lablesTrain[:train_subset])
    tf_valid_dataset = tf.constant(completeDataTrain[train_subset:20000,:])
    tf_test_dataset = tf.constant(completeDataTest[:train_subset,:])
    valid_labels = lablesTrain[train_subset:20000]
    test_labels = lablesTest[:train_subset]
    weights = tf.Variable(tf.truncated_normal([28,num_labels],dtype=tf.float32))
    biases = tf.Variable(tf.zeros([num_labels]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    
    train_prediction = tf.nn.sigmoid(logits)
    valid_prediction = tf.nn.sigmoid(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.sigmoid(tf.matmul(tf_test_dataset, weights) + biases)
    
num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

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
        