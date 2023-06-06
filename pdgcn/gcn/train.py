from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from models import GAutoencoder
from metrics import *

class Training():
  def __init__(self):
     self.model = GAutoencoder
     self.learning_rate = 0.001
     self.dropout = 0.1
     self.weight_decay = 5e-4
     self.early_stopping = 100
     self.max_degree = 3
     self.latent_factor = 25
     self.epochs = 200
        
  def train(self,train_arr, test_arr):
        # Settings
    
    # Load data
    adj, features, size_u, size_v, logits_train, logits_test, train_mask, test_mask, labels = load_data(train_arr, test_arr) 
    # Some preprocessing
    if self.model == 'GAutoencoder':
        model_func = GAutoencoder
    else:
        raise ValueError('Invalid argument for model: ' + str(self.model))
    
    
    # Define placeholders
    placeholders = {
        'adjacency_matrix': tf.compat.v1.placeholder(tf.float32, shape=adj.shape),
        'Feature_matrix': tf.compat.v1.placeholder(tf.float32, shape=features.shape),
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, logits_train.shape[1])),
        'labels_mask': tf.compat.v1.placeholder(tf.int32),
        'negative_mask': tf.compat.v1.placeholder(tf.int32)
    }
    
    # Create model
    model = model_func(placeholders, size_u, size_v, self.latent_factor_num)
    
    # Initialize session
    sess = tf.compat.v1.Session()
    
    
    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # Train model
    for epoch in range(self.epochs):

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy])

        print("Epoch:", '%04d' % (epoch + 1), 
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), 
              "time=", "{:.5f}".format(time.time()))
     
    print("Optimization Finished!")


    
