import tensorflow as tf
import os
import numpy as np
import utils
from input_data import get_pixel, one2batchbyRANDOMCROP

model_dir = "D:\\xunleiDownload\\RMB\\model"
#这个py文件中的投票方法只适用于RANDOM_CROP，不适用于RESIZE
model = os.path.join(model_dir, "random_crop")

test_dir = "D:\\xunleiDownload\\RMB\\public_test_data\\public_test_data"
test_label = "D:\\xunleiDownload\\RMB\\test_label.csv"

ckpt = tf.train.get_checkpoint_state(model_dir)

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

	#加载最新的模型参数
	lastest_model = ckpt.model_checkpoint_path
	print(lastest_model)

	#先导入结构，再把数据导入到会话中
	saver = tf.train.import_meta_graph(os.path.join(model_dir, "random_crop.meta"))
	saver.restore(sess, lastest_model)
	
	#从图中取出之前添加的训练句柄
	graph = tf.get_default_graph()
	outputs_op = tf.get_collection("outputs")[0]
	
	#输入模型中的数据入口
	X = graph.get_tensor_by_name("X:0")
	 
	#如果由原先重名的训练结果，就删掉，方便后面创造一个新的
	if os.path.exists(test_label):
		os.remove(test_label)

	f = open(test_label, "w")
	f.writelines(",".join(["name", "label"]) + "\n")
	print("Success to create csv.")

	#结果中的数字所代表的数字
	classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]

	for name, data in one2batchbyRANDOMCROP(test_dir):
		
		res = sess.run(outputs_op, feed_dict = {X: np.array(data)})
		index = sess.run(tf.argmax(res, 1))
		#print(index)
		index = most_in_index(index)
		f.writelines(",".join([name, classes[index]]) + "\n")

	f.close()
