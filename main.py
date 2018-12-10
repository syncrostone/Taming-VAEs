# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import random
import os
from scipy.misc import imsave

from model import VAE
from data_manager import DataManager

tf.app.flags.DEFINE_integer("epoch_size", 2000, "epoch size")
tf.app.flags.DEFINE_integer("batch_size", 64, "batch size")
tf.app.flags.DEFINE_float("beta", 100.0, "beta param for latent loss")
tf.app.flags.DEFINE_float("learning_rate", 1e-6, "learning rate")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints", "checkpoint directory")
tf.app.flags.DEFINE_string("log_file", "./log", "log file directory")
tf.app.flags.DEFINE_boolean("training", True, "training or not")
tf.app.flags.DEFINE_float("alpha", .99, "constraint moving average parameter")
tf.app.flags.DEFINE_float("kappa", 1, "tolerance hyper-parameter")
tf.app.flags.DEFINE_boolean("use_geco", True, "use geco fitter or use original fitter")
tf.app.flags.DEFINE_float("lagrange_mult_param", .00001, "multiplier by which to multiply exponentional of moving average when updating the lagrange multiplier")

flags = tf.app.flags.FLAGS

def train(sess,
          model,
          manager,
          saver):

  summary_writer = tf.summary.FileWriter(flags.log_file, sess.graph)
  
  n_samples = manager.sample_size

  reconstruct_check_images = manager.get_random_images(10)

  indices = list(range(n_samples))

  step = 0
  
  # Training cycle
  for epoch in range(flags.epoch_size):
    print("epoch: " + str(epoch))
    # Shuffle image indices
    random.shuffle(indices)
    
    avg_cost = 0.0
    total_batch = n_samples // flags.batch_size
    
    # Loop over all batches
    for i in range(total_batch):
      print("epoch: " + str(epoch) + "batch: " + str(i))
      # Generate image batch
      batch_indices = indices[flags.batch_size*i : flags.batch_size*(i+1)]
      batch_xs = manager.get_images(batch_indices)
      
      # Fit training using batch data
      reconstr_loss, latent_loss, summary_str = model.partial_fit(sess, batch_xs, step)
      summary_writer.add_summary(summary_str, step)
      step += 1

    # Image reconstruction check
    reconstruct_check(sess, model, reconstruct_check_images)

    # Disentangle check
    disentangle_check(sess, model, manager)

    # Save checkpoint
    saver.save(sess, flags.checkpoint_dir + '/' + 'checkpoint', global_step = step)

    
def reconstruct_check(sess, model, images):
  # Check image reconstruction
  x_reconstruct = model.reconstruct(sess, images)

  if not os.path.exists("reconstr_img"):
    os.mkdir("reconstr_img")

  for i in range(len(images)):
    org_img = images[i].reshape(64, 64)
    org_img = org_img.astype(np.float32)
    reconstr_img = x_reconstruct[i].reshape(64, 64)
    imsave("reconstr_img/org_{0}.png".format(i),      org_img)
    imsave("reconstr_img/reconstr_{0}.png".format(i), reconstr_img)


def disentangle_check(sess, model, manager, save_original=False):
  img = manager.get_image(shape=1, scale=2, orientation=5)
  if save_original:
    imsave("original.png", img.reshape(64, 64).astype(np.float32))
    
  batch_xs = [img]
  z_mean, z_log_sigma_sq = model.transform(sess, batch_xs)
  z_sigma_sq = np.exp(z_log_sigma_sq)[0]

  # Print variance
  zss_str = ""
  for i,zss in enumerate(z_sigma_sq):
    str = "z{0}={1:.4f}".format(i,zss)
    zss_str += str + ", "
  print(zss_str)

  # Save disentangled images
  z_m = z_mean[0]
  n_z = 10

  if not os.path.exists("disentangle_img"):
    os.mkdir("disentangle_img")

  for target_z_index in range(n_z):
    for ri in range(n_z):
      value = -3.0 + (6.0 / 9.0) * ri
      z_mean2 = np.zeros((1, n_z))
      for i in range(n_z):
        if( i == target_z_index ):
          z_mean2[0][i] = value
        else:
          z_mean2[0][i] = z_m[i]
      reconstr_img = model.generate(sess, z_mean2)
      rimg = reconstr_img[0].reshape(64, 64)
      imsave("disentangle_img/check_z{0}_{1}.png".format(target_z_index,ri), rimg)
      

def load_checkpoints(sess):
  saver = tf.train.Saver()
  checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("loaded checkpoint: {0}".format(checkpoint.model_checkpoint_path))
  else:
    print("Could not find old checkpoint")
    if not os.path.exists(flags.checkpoint_dir):
      os.mkdir(flags.checkpoint_dir)
  return saver


def main(argv):
  manager = DataManager()
  manager.load()

  sess = tf.Session()
  #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  
  model = VAE(beta=flags.beta,
              learning_rate=flags.learning_rate,
              alpha = flags.alpha,
              kappa = flags.kappa,
              lagrange_mult_param = flags.lagrange_mult_param,
              use_geco = flags.use_geco)
  
  sess.run(tf.global_variables_initializer())

  saver = load_checkpoints(sess)

  if flags.training:
    # Train
    train(sess, model, manager, saver)
  else:
    reconstruct_check_images = manager.get_random_images(10)
    # Image reconstruction check
    reconstruct_check(sess, model, reconstruct_check_images)
    # Disentangle check
    disentangle_check(sess, model, manager)
  

if __name__ == '__main__':
  tf.app.run()
