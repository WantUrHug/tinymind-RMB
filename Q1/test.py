import tensorflow as tf
import os, time
import numpy as np
import utils
from input_data import get_pixel, one2batchbyRANDOMCROP, batch2batchbyRANDOMCROP

model_dir = "D:\\xunleiDownload\\RMB\\GPUrandomcrop_model"
#这个py文件中的投票方法只适用于RANDOM_CROP，不适用于RESIZE!
#model = os.path.join(model_dir, "GPUrandomcrop_model")

test_dir = "D:\\xunleiDownload\\RMB\\public_test_data\\public_test_data"
test_label = "D:\\xunleiDownload\\RMB\\test_label1.csv"

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

with tf.Session(config=tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)) as sess:

	#加载最新的模型参数
	lastest_model = ckpt.model_checkpoint_path
	#print(lastest_model)

	#先导入结构，再把数据导入到会话中
	saver = tf.train.import_meta_graph(os.path.join(model_dir, "random_crop.meta"))
	saver.restore(sess, lastest_model)
	
	#从图中取出之前添加的训练句柄
	graph = tf.get_default_graph()
	outputs_op = tf.get_collection("outputs")[0]
	
	#输入模型中的数据入口
	X = graph.get_tensor_by_name("X:0")

	#结果中的数字所代表的数字
	classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]

	num = 0
	IMG_NUM = 2
	BATCH_SIZE = 10
	
	if os.path.exists(test_label):
		#如果文件已经存在，就需要判断其中是否有已经训练完的数据，数量是否够，如果
		#够就不再做重复的测试，如果不够就需要测试剩余的部分
		exist_csv = np.loadtxt(test_label, dtype = str, skiprows = (1), delimiter= ', ')
		label_num = len(exists)
		if label_num == 20000:
			raise ValueError("The number of data in this csv is 20000.No need to train more.")
		#已经存在并且没有完成，用a+表示续写
		f = open(test_label, "a+")
		print("Continue to finish csv.")
	else:
		f = open(test_label, "w")
		f.writelines(",".join(["name", "label"]) + "\n")
		print("Success to create csv.")

	start_time = time.time()
	for name_list, data in batch2batchbyRANDOMCROP(test_dir, img_num = IMG_NUM, batch_size = BATCH_SIZE):
		#print(len(name_list))
		res = sess.run(outputs_op, feed_dict = {X: data})
		result_size = len(name_list)
		final = sess.run(tf.argmax(res, 1))
		print(final)
		for i in range(result_size):
			#index = sess.run(tf.argmax(res[i*BATCH_SIZE: (i+1)*BATCH_SIZE], 1))
			#print(index.shape)
			#print(index)
			id = most_in_index(final[i*BATCH_SIZE: (i+1)*BATCH_SIZE])
			print(id)
			f.writelines(",".join([name_list[i], classes[id]]) + "\n")
			if num%500 == 0:
				print("Had processing %d pictures"%num, end = ".")
				end_time = time.time()
				print("Take %.1f second per image for batch size %s"%((end_time-start_time)/result_size, 10))

	f.close()
