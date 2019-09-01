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
TEST_BATCH = 100
MAXSTEP = 2000
CHECK_STEP = 20
SAVE_STEP = 200
NUM_CLASSES = 10

train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv"
model_dir = "D:\\xunleiDownload\\RMB\\model"

#test_model_dir = "D:\\GitFile\\RMB\\Q1\\model"

#if not os.path.exists(model_dir):
#	raise ValueError("Could't find model dir where saved the model.")


#重新准备数据，一模一样的方法
train_dataset = tf.data.Dataset.from_generator(train_data_gen, output_types = (tf.float32, tf.int32))
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE).repeat()
train_iterator = train_dataset.make_one_shot_iterator()

next_train_data, next_train_label = train_iterator.get_next()

test_dataset = tf.data.Dataset.from_generator(test_data_gen, output_types = (tf.float32, tf.int32))
test_dataset = test_dataset.batch(100).repeat()
test_iterator = test_dataset.make_one_shot_iterator()

next_test_data, next_test_label = test_iterator.get_next()

print("Finish preparing data...")

history = {}
history["train_loss"] = []
history["test_loss"] = []
history["train_acc"] = []
history["test_acc"] = []

ckpt = tf.train.get_checkpoint_state(model_dir)
#print(ckpt)

with tf.Session() as sess:

	lastest_model = ckpt.model_checkpoint_path

	for i in range(len(lastest_model)):
		if lastest_model[i] == "-":
			print("The model has been trained for %s steps." % lastest_model[i + 1:])

	saver = tf.train.import_meta_graph(os.path.join(model_dir, "random_crop.meta"))
	saver.restore(sess, lastest_model)

	graph = tf.get_default_graph()
	
	#train_op = graph.get_tensor_by_name("train_op:0")
	#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      print(train_op)
	train_op = tf.get_collection("train_op")[0]
	#
	print(train_op)

	