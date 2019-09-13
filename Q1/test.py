import tensorflow as tf
import os, time
import numpy as np
import utils
from input_data import get_pixel, one2batchbyRANDOMCROP, batch2batchbyRANDOMCROP

model_dir = "D:\\xunleiDownload\\RMB\\GPUrandomcrop_model"
#这个py文件中的投票方法只适用于RANDOM_CROP，不适用于RESIZE!
#model = os.path.join(model_dir, "GPUrandomcrop_model")

test_dir = "D:\\xunleiDownload\\RMB\\public_test_data\\public_test_data"
test_label = "D:\\xunleiDownload\\RMB\\test_label2.csv"

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
	IMG_NUM = 4
	BATCH_SIZE = 10
	
	if os.path.exists(test_label):
		#如果文件已经存在，就需要判断其中是否有已经训练完的数据，数量是否够，如果
		#够就不再做重复的测试，如果不够就需要测试剩余的部分
		exist_csv = np.loadtxt(test_label, dtype = str, skiprows = (1), delimiter= ', ')
		label_num = len(exist_csv)
		#需要跳过的次数
		jump_num = int(label_num/IMG_NUM)
		#之后在循环中判断已经跳过的轮数
		jump_flag = 0
		if label_num == 20000:
			raise ValueError("The number of data in this csv is 20000.No need to train more.")
		del exist_csv
		#已经存在并且没有完成，用a+表示续写
		#在utils.py中定义一个csvHelper，帮助我们封装一些自己的方法，处理起来更简单
		helper = utils.csvHelper(test_label)
		print("Continue to finish csv.")
	else:
		helper = utils.csvHelper(test_label)
		helper.writelines(",".join(["name", "label"]) + "\n", "w")
		#f = open(test_label, "w")
		#f.writelines(",".join(["name", "label"]) + "\n")
		print("Success to create csv and title.")

	start_time = time.time()
	for name_list, data in batch2batchbyRANDOMCROP(test_dir, img_num = IMG_NUM, batch_size = BATCH_SIZE):
		#print(len(name_list))
		if jump_flag < jump_num:
			jump_flag += 1
			continue

		res = sess.run(outputs_op, feed_dict = {X: data})
		result_size = len(name_list)
		final = sess.run(tf.argmax(res, 1))
		
		#之前的做法是每处理一张图片就往csv中插入一行，效率较低，而且可能造成一个问题，就是中断的时候正好没有写完数据
		#那么下次重写的时候会遇到不整的问题。例如，我们每次处理4张图片，如果是在第三张中断，那么csv中就会有0.1.2的记录，
		#但是我们写的迭代器 batch2batchbyRANDOMCROP，每次读取四张，跳过也是四张四张，就意味着我们会忽略了第四张，处理的
		#结果会是0.1.2.4.5.6....所以每次张写入一下数据，哪怕是自行中断错误也较低可能出现这种情况。
		char = ""
		for i in range(result_size):

			id = most_in_index(final[i*BATCH_SIZE: (i+1)*BATCH_SIZE])
			char += ",".join([name_list[i], classes[id]])
			char += "\n"

		#把四次的结果一次性写入
		helper.writelines(char)
		num += result_size
		if num%500 == 0:
			print("Had processing %d pictures, %d to %d"%(num, label_num, label_num + 500), end = ".")
			label_num += 500
			end_time = time.time()
			print("Take %.2f second per image for batch size %s."%((end_time-start_time)/500, 10))
			#之前忘了重置 start_time,发现时间越来越长，几乎是线性的.
			start_time = time.time()
