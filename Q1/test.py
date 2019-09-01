import tensorflow as tf
import os
import numpy as np
import utils
from input_data import get_pixel, one2batchbyRANDOMCROP

model_dir = "D:\\xunleiDownload\\RMB\\model"
model = os.path.join(model_dir, "random_crop")

test_dir = "D:\\xunleiDownload\\RMB\\public_test_data\\public_test_data"
test_label = "D:\\xunleiDownload\\RMB\\test_label.csv"


ckpt = tf.train.get_checkpoint_state(model_dir)
print(ckpt)
#global_step = tf.Variable(0, trainable = False)

def most_in_index(index):
	'''
	对投票结果进行判断，例如index有10个数字，我们找出其中出现最多的数字，作为
	最终的结果.
	'''
	ind = [0 for i in range(10)]
	for i in range(10):
		ind[index[i]] += 1
	max = 0
	for i in range(1, 10):
		if ind[i] > ind[max]:
			max = i
	return max

with tf.Session() as sess:

	lastest_model = ckpt.model_checkpoint_path
	print(lastest_model)

	saver = tf.train.import_meta_graph(os.path.join(model_dir, "random_crop.meta"))

	saver.restore(sess, lastest_model)

	graph = tf.get_default_graph()

	outputs_op = tf.get_collection("outputs")[0]
	print(outputs_op)
	X = graph.get_tensor_by_name("X:0")
	print(X)

	#print(graph)
	t = 0
	f = open(test_label, "w")
	f.writelines(",".join(["name", "label"]) + "\n")
	print("Finish to create csv.")

	
	classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]

	for i, j in one2batchbyRANDOMCROP(test_dir):
		
		res = sess.run(outputs_op, feed_dict = {X: np.array(j)})
		index = sess.run(tf.argmax(res, 0))
		#print(index)
		index = most_in_index(index)
		f.writelines(",".join([i, classes[index]]) + "\n")

	f.close()
