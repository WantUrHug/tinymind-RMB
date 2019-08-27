import tensorflow as tf
import time
import os
import model
from input_data import train_data_gen, test_data_gen
from utils import show_result

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BATCH_SIZE = 128
IMAGE_H = 224
IMAGE_W = 224
CHANNEL = 3
TEST_NUM = 2000
TEST_BATCH = 100
MAXSTEP = 1000
CHECK_STEP = 50
NUM_CLASSES = 10

train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv"
model_dir = "D:\\GitFile\\RMB\\Q1\\model"

train_dataset = tf.data.Dataset.from_generator(train_data_gen, output_types = (tf.float32, tf.int32))
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE).repeat()
train_iterator = train_dataset.make_one_shot_iterator()

next_train_data, next_train_label = train_iterator.get_next()

test_dataset = tf.data.Dataset.from_generator(test_data_gen, output_types = (tf.float32, tf.int32))
test_dataset = test_dataset.batch(100).repeat()
test_iterator = test_dataset.make_one_shot_iterator()

next_test_data, next_test_label = test_iterator.get_next()

print("Finish preparing data...")

X = tf.placeholder(tf.float32, [None, IMAGE_H, IMAGE_W, CHANNEL])
Y = tf.placeholder(tf.int32, [None, NUM_CLASSES])

outputs = model.inference(X, 10)

loss_op = model.loss(logits = outputs, labels = Y)
train_op = model.train(loss = loss_op, lr = 0.001)
acc_op = model.evaluation(logits = outputs, labels = Y)

history = {}
history["train_loss"] = []
history["test_loss"] = []
history["train_acc"] = []
history["test_acc"] = []

with tf.Session() as sess:

	saver = tf.train.Saver()

	sess.run(tf.global_variables_initializer())
	time_cost = 0
	start_time = time.time()

	for step in range(1, MAXSTEP + 1):

		train_data, train_labels = sess.run((next_train_data, next_train_label))
		sess.run(train_op,  feed_dict = {X: train_data, Y: train_labels})

		if step%CHECK_STEP == 0:
			train_loss, train_acc = sess.run((loss_op, acc_op), feed_dict = {X: train_data, Y: train_labels})
			history["train_loss"].append(train_loss)
			history["train_acc"].append(train_acc)


			test_data, test_labels = sess.run((next_test_data, next_test_label))
			test_loss, test_acc = sess.run((loss_op, acc_op), feed_dict = {X: test_data, Y: test_labels})
			history["test_loss"].append(test_loss)
			history["test_acc"].append(test_acc)

			print("Step %d, train loss: %.2f, train acc: %.2f%%, test loss: %.2f, test_acc: %.2f%%." % (step, train_loss, train_acc*100, test_loss, test_acc*100))

	saver.save(sess, os.path.join(model_dir, "random_crop.ckpt"))

show_result(history, )