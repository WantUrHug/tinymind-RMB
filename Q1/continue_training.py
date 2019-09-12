import tensorflow as tf
import time, os
import model
from input_data import train_data_gen, test_data_gen
from utils import show_result

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BATCH_SIZE = 64
#TEST_BATCH = 100
MAXSTEP = 100
CHECK_STEP = 25
SAVE_STEP = 100

train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv"
#model_dir = "D:\\xunleiDownload\\RMB\\resize_model"
model_dir = "D:\\xunleiDownload\\RMB\\GPUrandomcrop_model"

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

#暂时不考虑记录训练的情况，一般训练起来时间长一些肯定是好的
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

with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:

	lastest_model = ckpt.model_checkpoint_path

	for i in range(len(lastest_model)):
		if lastest_model[i] == "-":
			ran = int(lastest_model[i+1:])
			print("The model has been trained for %s steps." % ran)
			

	saver = tf.train.import_meta_graph(os.path.join(model_dir, "random_crop.meta"))
	saver.restore(sess, lastest_model)

	graph = tf.get_default_graph()
	
	#取出训练句柄和原先的占位符，因为训练需要这三者                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    print(train_op)
	train_op = tf.get_collection("train_op")[0]
	#print(train_op)
	X = graph.get_tensor_by_name("X:0")
	Y = graph.get_tensor_by_name("Y:0")
	#print(X, Y)
	#即使是继续训练最好也是要可以观察进展
	loss_op = tf.get_collection("loss_op")[0]
	acc_op = tf.get_collection("acc_op")[0]

	start_time = time.time()
	for step in range(1, MAXSTEP + 1):

		#继续训练
		train_data, train_labels = sess.run((next_train_data, next_train_label))
		sess.run(train_op,  feed_dict = {X: train_data, Y: train_labels})

		#按照设定的步长周期性地计算训练和验证的情况
		if step%CHECK_STEP == 0:

			train_loss, train_acc = sess.run((loss_op, acc_op), feed_dict = {X: train_data, Y: train_labels})
		
			test_data, test_labels = sess.run((next_test_data, next_test_label))
			test_loss, test_acc = sess.run((loss_op, acc_op), feed_dict = {X: test_data, Y: test_labels})
			
			time_cost = time.time() - start_time
			
			print("Step %d, time cost: %.1fs, train loss: %.2f, train acc: %.2f%%, test loss: %.2f, test_acc: %.2f%%." % (step+ran, time_cost, train_loss, train_acc*100, test_loss, test_acc*100))
			#print("Step %d, time cost: %.1fs, train loss: %.2f, train acc: %.2f%%." % (step, time_cost, train_loss, train_acc*100))
			start_time = time.time()

		#按照设定的步长去周期性地保存模型
		if step%SAVE_STEP == 0:
			saver.save(sess, os.path.join(model_dir, "random_crop"), global_step = step + ran, write_meta_graph = False)