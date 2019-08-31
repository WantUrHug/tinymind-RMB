import tensorflow as tf
import os
import numpy as np
import utils
from input_data import get_pixel, one2batchbyRANDOMCROP

model_dir = "D:\\GitFile\\RMB\\Q1\\model"
model = os.path.join(model_dir, "random_crop")

test_dir = "D:\\xunleiDownload\\RMB\\public_test_data\\public_test_data"

ckpt = tf.train.get_checkpoint_state(model_dir)
print(ckpt)
#global_step = tf.Variable(0, trainable = False)

with tf.Session() as sess:

	lastest_model = ckpt.model_checkpoint_path

	saver = tf.train.import_meta_graph(os.path.join(model_dir, lastest_model+".meta"))

	saver.restore(sess, lastest_model)

	graph = tf.get_default_graph()

	outputs_op = tf.get_collection("outputs")[0]
	print(outputs_op)
	X = graph.get_tensor_by_name("X:0")
	print(X)

	#print(graph)
	t = 0
	for i, j in one2batchbyRANDOMCROP(test_dir):
		print(i)
		res = sess.run(outputs_op, feed_dict = {X: np.array(j)})
		index = sess.run(tf.argmax(res, 0))
		print(index)
		t += 1
		if t > 10:
			break