import tensorflow as tf
import os

model_dir = "D:\\GitFile\\RMB\\Q1\\model"
model = os.path.join(model_dir, "random_crop.ckpt")

saver = tf.train.Saver()

with tf.Session() as sess:

