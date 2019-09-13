import tensorflow as tf
import time
import os
import model
from input_data import train_data_gen, test_data_gen
from utils import show_result, save_his_csv

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
#定义常数
BATCH_SIZE = 64
IMAGE_H = 224
IMAGE_W = 224
CHANNEL = 3
TEST_BATCH = 100
MAXSTEP = 500
CHECK_STEP = 20
SAVE_STEP = 100
NUM_CLASSES = 10

train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv"
#创建一个新的文件夹来存放使用GPU训练的模型
model_dir = "D:\\xunleiDownload\\RMB\\GPUrandomcrop_model"
#文件夹不存在时就需要创建文件夹
if not os.path.exists(model_dir):
	os.mkdir(model_dir)

train_dataset = tf.data.Dataset.from_generator(train_data_gen, output_types = (tf.float32, tf.int32))
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE).repeat()
train_iterator = train_dataset.make_one_shot_iterator()

next_train_data, next_train_label = train_iterator.get_next()

test_dataset = tf.data.Dataset.from_generator(test_data_gen, output_types = (tf.float32, tf.int32))
test_dataset = test_dataset.batch(100).repeat()
test_iterator = test_dataset.make_one_shot_iterator()

next_test_data, next_test_label = test_iterator.get_next()

print("Finish preparing data...")

#凡是变量都要添加名字，以便之后在保存的模型恢复时调用
X = tf.placeholder(tf.float32, [None, IMAGE_H, IMAGE_W, CHANNEL], name = "X")
#tf.add_to_collection("X", X)
Y = tf.placeholder(tf.int32, [None, NUM_CLASSES], name = "Y")

#定义一个操作添加到默认图之中，后续可以测试
outputs = model.Simple_VGG_19(X, 10)
tf.add_to_collection("outputs", outputs)

#各种句柄都要添加到图中
loss_op = model.loss(logits = outputs, labels = Y)
train_op = model.train(loss = loss_op, lr = 0.001)
acc_op = model.evaluation(logits = outputs, labels = Y)

tf.add_to_collection("train_op", train_op)
tf.add_to_collection("loss_op", loss_op)
tf.add_to_collection("acc_op", acc_op)


history = {}
history["train_loss"] = []
history["test_loss"] = []
history["train_acc"] = []
history["test_acc"] = []


with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:

	saver = tf.train.Saver(max_to_keep = 3)

	sess.run(tf.global_variables_initializer())
	
	tf.train.Saver().save(sess, os.path.join(model_dir, "random_crop"))

	time_cost = 0
	start_time = time.time()

	for step in range(1, MAXSTEP + 1):

		train_data, train_labels = sess.run((next_train_data, next_train_label))
		sess.run(train_op,  feed_dict = {X: train_data, Y: train_labels})

		if step%CHECK_STEP == 0:

			train_loss, train_acc = sess.run((loss_op, acc_op), feed_dict = {X: train_data, Y: train_labels})
		
			test_data, test_labels = sess.run((next_test_data, next_test_label))
			test_loss, test_acc = sess.run((loss_op, acc_op), feed_dict = {X: test_data, Y: test_labels})
			
			history["train_loss"].append(train_loss)
			history["train_acc"].append(train_acc)
			history["test_loss"].append(test_loss)
			history["test_acc"].append(test_acc)
			
			time_cost = time.time() - start_time
			
			print("Step %d, time cost: %.1fs, train loss: %.2f, train acc: %.2f%%, test loss: %.2f, test_acc: %.2f%%." % (step, time_cost, train_loss, train_acc*100, test_loss, test_acc*100))
			#print("Step %d, time cost: %.1fs, train loss: %.2f, train acc: %.2f%%." % (step, time_cost, train_loss, train_acc*100))
			start_time = time.time()

		if step%SAVE_STEP == 0:
			saver.save(sess, os.path.join(model_dir, "random_crop"), global_step = step, write_meta_graph = False)			

	#saver.save(sess, os.path.join(model_dir, "random_crop"), global_step = MAXSTEP)

show_result(history)
save_his_csv(history, '1.csv')