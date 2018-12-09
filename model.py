# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import tensorflow as tf


def fc_initializer(input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
  def _initializer(shape, dtype=dtype, partition_info=None):
    d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
    return tf.random_uniform(shape, minval=-d, maxval=d)
  return _initializer


class VAE(object):
  """ Beta Variational Auto Encoder."""
  
  def __init__(self,
               beta=100.0,
               learning_rate=1e-6,
               alpha = 0.99,
               kappa = 0.1,
               lagrange_mult_param = .5,
               use_geco = True):
    self.beta = beta
    self.use_geco = use_geco
    self.learning_rate = learning_rate
    self.alpha = alpha
    self.kappa = kappa
    self.lagrange_mult_param = .5

    # Create autoencoder network
    self._create_network()
    
    # Define loss function and corresponding optimizer
    self._create_loss_optimizer()

  def _conv2d_weight_variable(self, weight_shape, name, deconv=False):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
      input_channels  = weight_shape[3]
      output_channels = weight_shape[2]
    else:
      input_channels  = weight_shape[2]
      output_channels = weight_shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape,
                             initializer=conv_initializer(w, h, input_channels))
    bias   = tf.get_variable(name_b, bias_shape,
                             initializer=conv_initializer(w, h, input_channels))
    return weight, bias


  def _fc_weight_variable(self, weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)
    
    input_channels  = weight_shape[0]
    output_channels = weight_shape[1]
    d = 1.0 / np.sqrt(input_channels)
    bias_shape = [output_channels]

    weight = tf.get_variable(name_w, weight_shape, initializer=fc_initializer(input_channels))
    bias   = tf.get_variable(name_b, bias_shape,   initializer=fc_initializer(input_channels))
    return weight, bias
  
  
  def _get_deconv2d_output_size(self, input_height, input_width, filter_height,
                                filter_width, row_stride, col_stride, padding_type):
    if padding_type == 'VALID':
      out_height = (input_height - 1) * row_stride + filter_height
      out_width  = (input_width  - 1) * col_stride + filter_width
    elif padding_type == 'SAME':
      out_height = input_height * row_stride
      out_width  = input_width * col_stride
    return out_height, out_width
  
  
  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1],
                        padding='SAME')
  
  
  def _deconv2d(self, x, W, input_width, input_height, stride):
    filter_height = W.get_shape()[0].value
    filter_width  = W.get_shape()[1].value
    out_channel   = W.get_shape()[2].value
    
    out_height, out_width = self._get_deconv2d_output_size(input_height,
                                                           input_width,
                                                           filter_height,
                                                           filter_width,
                                                           stride,
                                                           stride,
                                                           'SAME')
    batch_size = tf.shape(x)[0]
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, stride, stride, 1],
                                  padding='SAME')

  def _sample_z(self, z_mean, z_log_sigma_sq):
    eps_shape = tf.shape(z_mean)
    eps = tf.random_normal( eps_shape, 0, 1, dtype=tf.float32 )
    # z = mu + sigma * epsilon
    z = tf.add(z_mean,
               tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z
    
  
  def _create_recognition_network(self, x, reuse=False):
    with tf.variable_scope("rec", reuse=reuse) as scope:
      # [filter_height, filter_width, in_channels, out_channels]
      W_conv1, b_conv1 = self._conv2d_weight_variable([4, 4, 1,  32], "conv1")
      W_conv2, b_conv2 = self._conv2d_weight_variable([4, 4, 32, 32], "conv2")
      W_conv3, b_conv3 = self._conv2d_weight_variable([4, 4, 32, 32], "conv3")
      W_conv4, b_conv4 = self._conv2d_weight_variable([4, 4, 32, 32], "conv4")
      W_fc1, b_fc1     = self._fc_weight_variable([4*4*32, 256], "fc1")
      W_fc2, b_fc2     = self._fc_weight_variable([256, 256], "fc2")
      W_fc3, b_fc3     = self._fc_weight_variable([256, 10],  "fc3")
      W_fc4, b_fc4     = self._fc_weight_variable([256, 10],  "fc4")

      x_reshaped = tf.reshape(x, [-1, 64, 64, 1])
      h_conv1 = tf.nn.relu(self._conv2d(x_reshaped, W_conv1, 2) + b_conv1) # (32, 32)
      h_conv2 = tf.nn.relu(self._conv2d(h_conv1,    W_conv2, 2) + b_conv2) # (16, 16)
      h_conv3 = tf.nn.relu(self._conv2d(h_conv2,    W_conv3, 2) + b_conv3) # (8, 8)
      h_conv4 = tf.nn.relu(self._conv2d(h_conv3,    W_conv4, 2) + b_conv4) # (4, 4)
      h_conv4_flat = tf.reshape(h_conv4, [-1, 4*4*32])
      h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1) + b_fc1)
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1,        W_fc2) + b_fc2)
      z_mean         = tf.matmul(h_fc2, W_fc3) + b_fc3
      z_log_sigma_sq = tf.matmul(h_fc2, W_fc4) + b_fc4
      return (z_mean, z_log_sigma_sq)

  
  def _create_generator_network(self, z, reuse=False):
    with tf.variable_scope("gen", reuse=reuse) as scope:
      W_fc1, b_fc1 = self._fc_weight_variable([10,  256],    "fc1")
      W_fc2, b_fc2 = self._fc_weight_variable([256, 4*4*32], "fc2")

      # [filter_height, filter_width, output_channels, in_channels]
      W_deconv1, b_deconv1 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv1", deconv=True)
      W_deconv2, b_deconv2 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv2", deconv=True)
      W_deconv3, b_deconv3 = self._conv2d_weight_variable([4, 4, 32, 32], "deconv3", deconv=True)
      W_deconv4, b_deconv4 = self._conv2d_weight_variable([4, 4,  1, 32], "deconv4", deconv=True)

      h_fc1 = tf.nn.relu(tf.matmul(z,     W_fc1) + b_fc1)
      h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
      h_fc2_reshaped = tf.reshape(h_fc2, [-1, 4, 4, 32])
      h_deconv1   = tf.nn.relu(self._deconv2d(h_fc2_reshaped, W_deconv1,  4,  4, 2) + b_deconv1)
      h_deconv2   = tf.nn.relu(self._deconv2d(h_deconv1,      W_deconv2,  8,  8, 2) + b_deconv2)
      h_deconv3   = tf.nn.relu(self._deconv2d(h_deconv2,      W_deconv3, 16, 16, 2) + b_deconv3)
      h_deconv4   =            self._deconv2d(h_deconv3,      W_deconv4, 32, 32, 2) + b_deconv4
      
      x_out_logit = tf.reshape(h_deconv4, [-1, 64*64*1])
      return x_out_logit

    
  def _create_network(self):
    # tf Graph input
    self.x = tf.placeholder(tf.float32, shape=[None, 4096])
    
    with tf.variable_scope("vae"):
      self.z_mean, self.z_log_sigma_sq = self._create_recognition_network(self.x)

      # Draw one sample z from Gaussian distribution
      # z = mu + sigma * epsilon
      self.z = self._sample_z(self.z_mean, self.z_log_sigma_sq)

      self.x_out_logit = self._create_generator_network(self.z)
      self.x_out = tf.nn.sigmoid(self.x_out_logit)
      
      
  def _create_loss_optimizer(self):
    self.lagrange = tf.Variable(1/self.beta, trainable=False)
    self.lagrange = tf.Print(self.lagrange, [self.lagrange],"in graph, value of lagrange mult is: ")
    with tf.variable_scope("opt"):
      if not self.use_geco:
        # Reconstruction loss
        reconstr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x,
                                                                logits=self.x_out_logit)
        reconstr_loss = tf.reduce_sum(reconstr_loss, 1)
        self.reconstr_loss = tf.reduce_mean(reconstr_loss)

        # Latent loss
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.latent_loss = tf.reduce_mean(latent_loss)
        
        # Loss term with beta scaling
        self.loss = self.reconstr_loss + self.beta* tf.abs(self.latent_loss)
      if self.use_geco:
        #Uses reconstruction error here.
        self.reconstr_loss =tf.reduce_mean((tf.reduce_sum(tf.square(self.x - self.x_out), 1) - self.kappa**2))
        
        #KL divergence
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.lagrange = self.lagrange_mult
        #Loss term with lagrange scaling
        self.loss = self.latent_loss + self.lagrange * self.reconstr_loss
      
      reconstr_loss_summary_op = tf.summary.scalar('reconstr_loss', self.reconstr_loss)
      latent_loss_summary_op   = tf.summary.scalar('latent_loss',   self.latent_loss)
      self.summary_op = tf.summary.merge([reconstr_loss_summary_op, latent_loss_summary_op])
      self.optimizer = tf.train.AdamOptimizer(
        learning_rate=self.learning_rate).minimize(self.loss)


  def reconstruction_error(self, sess, xs):
    """Calculate reconstruction error as defined on page 8 of Taming VAEs"""
    #z_mean, z_log_sigma_sq = self.transform(sess,xs)
    #z_sample = self._sample_z(z_mean, z_log_sigma_sq)

    reconstructed_z = sess.run(self.x_out, feed_dict = {self.x:xs})
    return (tf.reduce_mean(tf.reduce_sum(tf.square(xs - reconstructed_z), 1) - self.kappa**2))
    #reconstructed_z = self.generate(sess, z_sample)
    # return (tf.reduce_sum(tf.square(xs - reconstructed_z)) - self.kappa**2)

  def geco_partial_fit(self, sess, xs, step):
    """Train model as described in Taming VAEs."""
    C_curr = self.reconstruction_error(sess, xs)
    print("early C_curr" + str(C_curr.eval(session=sess)))
    if step == 0:
      print("step is 0, initializing")
      self.lagrange_mult = 1.0/self.beta
      self.C_ma = C_curr
    else:
      self.C_ma = self.alpha * self.C_ma + (1 - self.alpha)*C_curr
    C_curr = C_curr + tf.stop_gradient(self.C_ma - C_curr)
    print("C_curr" + str(C_curr.eval(session=sess)))
    self.lagrange_mult = self.lagrange_mult * tf.exp(self.lagrange_mult_param * C_curr)

    print("lagrange mult: " + str(self.lagrange_mult.eval(session=sess)))
    self.optimizer = sess.run((self.optimizer),
                              feed_dict = {
                                self.x : xs,
                                self.lagrange : self.lagrange_mult.eval(session = sess)
                              })
    _, reconstr_loss, latent_loss, summary_str = sess.run((self.optimizer,
                                                           self.reconstr_loss,
                                                           self.latent_loss,
                                                           self.summary_op),
                                                          feed_dict={
                                                            self.x : xs,
                                                            self.lagrange : self.lagrange_mult.eval(session = sess)
                                                          })
    return reconstr_loss, latent_loss, summary_str
    
  def partial_fit(self, sess, xs, step):
    """Train model based on mini-batch of input data.
    
    Return loss of mini-batch.
    """
    if not self.use_geco:
      _, reconstr_loss, latent_loss, summary_str = sess.run((self.optimizer,
                                                             self.reconstr_loss,
                                                             self.latent_loss,
                                                             self.summary_op),
                                                            feed_dict={
                                                              self.x : xs,
                                                              self.lagrange_mult: self.lagrange_mult
                                                            })
      return reconstr_loss, latent_loss, summary_str
    if self.use_geco:
      return self.geco_partial_fit(sess, xs, step)

  def reconstruct(self, sess, xs):
    """ Reconstruct given data. """
    # Original VAE output
    return sess.run(self.x_out, 
                    feed_dict={self.x: xs})

  
  def transform(self, sess, xs):
    """Transform data by mapping it into the latent space."""
    return sess.run( [self.z_mean, self.z_log_sigma_sq],
                     feed_dict={self.x: xs} )
  

  def generate(self, sess, zs):
    """ Generate data by sampling from latent space. """
    return sess.run( self.x_out,
                     feed_dict={self.z: zs} )
