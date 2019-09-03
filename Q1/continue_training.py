import tensorflow as tf
import time
import os
import model
from input_data import train_data_gen, test_data_gen
from utils import show_result

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BATCH_SIZE = 128+1
#TEST_BATCH = 100
MAXSTEP = 2000
CHECK_STEP = 20
SAVE_STEP = 200

train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv"
model_dir = "D:\\xunleiDownload\\RMB\\resize_model"

#test_model_dir = "D:\\GitFile\\RMB\\Q1\\model"
#if not os.path.exists(model_dir):
#	raise ValueError("Could't find model dir where saved the model.")


#重新准备数据，一模一样的方法.先前似乎看有人说，batch_size不能更改，但是我试了似乎没问题
train_dataset = tf.data.Dataset.from_generator(train_data_gen, output_types = (tf.float32, tf.int32))
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE).repeat()
train_iterator = train_dataset.make_one_shot_iterator()

next_train_data, next_train_label = train_iterator.get_next()

test_dataset = tf.data.Dataset.from_generator(test_data_gen, output_types = (tf.float32, tf.int32))
test_dataset = test_dataset.batch(100).repeat()
test_iterator = test_dataset.make_one_shot_iterator()

next_test_data, next_test_label = test_iterator.get_next()

print("Finish preparing data...")

#history = {}
#history["train_loss"] = []
#history["test_loss"] = []
#history["train_acc"] = []
#history["test_acc"] = []


#目前运行的时候程序会覆盖掉原本的chechpoint记录，重新记录，所以
#要记得加上原先已经训练好的步数，或者是重新开一个文件夹，来存放新的
#训练结果，就不会改变原先的训练记录，不过要注意我现在是在原文件夹中做，
#所以meta文件还在，如果是在新的文件夹中就需要保存新的meta文件.
#但缺点就是会造成大量冗余的数据.
ckpt = tf.train.get_checkpoint_state(model_dir)

with tf.Session() as sess:

	lastest_model = ckpt.model_checkpoint_path

	#ran = ""
	for i in range(len(lastest_model)):
		if lastest_model[i] == "-":
			print("The model has been trained for %s steps." % lastest_model[i + 1:])
			ran = int(lastest_model[i+1:])

	saver = tf.train.import_meta_graph(os.path.join(model_dir, "resize.meta"))
	saver.restore(sess, lastest_model)

	graph = tf.get_default_graph()
	
	#取出训练句柄和原先的占位符，因为训练需要这三者                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    print(train_op)
	train_op = tf.get_collection("train_op")[0]
	#print(train_op)
	X = graph.get_tensor_by_name("X:0")
	Y = graph.get_tensor_by_name("Y:0")
	#print(X, Y)
	
	for step in range(1, MAXSTEP + 1):

		#继续训练
		train_data, train_labels = sess.run((next_train_data, next_train_label))
		sess.run(train_op,  feed_dict = {X: train_data, Y: train_labels})

		#按照设定的步长去周期性地保存模型
		if i%SAVE_STEP == 0:
			saver.save(sess, os.path.join(model_dir, "resize"), global_step = step + ran, write_meta_graph = False)