import tensorflow as tf
import os
import cv2
import time
from input_data import get_file

BATCH_SIZE = 128
IMAGE_H = 224
IMAGE_W = 224
CHANNEL = 3

train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv"

classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
NUM_CLASSES = len(classes)
class_dict = {}
for i, cls in enumerate(classes):
	class_dict[cls] = [1 if j == i else 0 for j in range(NUM_CLASSES)]

data_list, label_list = get_file(train_dir, label_csv, class_dict)
print("Finish prepareing original data...")

def args_wrapper(*args, **kwargs):
	def outer_wrapper(fun):
		def inner_wrapper():
			return fun(*args, **kwargs)
		return inner_wrapper
	return outer_wrapper

@args_wrapper(xs = data_list, labels = label_list, image_h = IMAGE_H, image_w = IMAGE_W)
def data_gen(xs, labels, image_h, image_w):
	
	for x, label in zip(xs, labels):

		data = cv2.imread(x)

		#regularization.
		yield data/255, label

train_dataset = tf.data.Dataset.from_generator(data_gen, output_types = (tf.float32, tf.int32))
train_dataset = train_dataset.shuffle(100).batch(BATCH_SIZE).repeat()
train_iterator = train_dataset.make_one_shot_iterator()

next_data, next_label = train_iterator.get_next()

X = tf.placeholder(tf.float32, [None, IMAGE_H, IMAGE_W, CHANNEL])
Y = tf.placeholder(tf.int32, [None, NUM_CLASSES])

#build model
def inference(inputs):

	#inputs.shape = [None, IMAGE_H, IMAGE_W, CHANNEL]
	outputs = tf.layers.Conv2D(filters = 16, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(inputs)
	#shape = [None, IMAGE_H, IMAGE_W, 16]
	outputs = tf.layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#shape = [None, IMAGE_H, IMAGE_W, 32]
	outputs = tf.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)
	#shape = [None, IMAGE_H/2, IMAGE_W/2, 32]
	outputs = tf.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#shape = [None, IMAGE_H/2, IMAGE_W/2, 64]
	outputs = tf.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#shape = [None, IMAGE_H/2, IMAGE_W/2, 128]
	outputs = tf.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)
	#shape = [None, IMAGE_H/4, IMAGE_W/4, 128]
	outputs = tf.layers.Flatten()(outputs)
	#shape = [None, IAMGE_H*IMAGE_W*8]
	outputs = tf.layers.Dense(512, activation = "relu")(outputs)
	outputs = tf.layers.Dense(512, activation = "relu")(outputs)
	outputs = tf.layers.Dense(NUM_CLASSES, activation = "relu")(outputs)

	return outputs

outputs = inference(X)