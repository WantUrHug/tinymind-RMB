import tensorflow as tf
from tensorflow.layers import Conv2D, MaxPooling2D, Flatten, Dense

def inference(inputs, num_classes):
	#太深的网络定义了电脑跑不动，难受
	#这是用tf.layers中定义的一些api来创建网络，忽略了一些细节，但是可以
	#更快的搭建起网络，很方便测试

	outputs = tf.layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(inputs)

	outputs = tf.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)
	
	outputs = tf.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	
	outputs = tf.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)
	
	outputs = tf.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	
	outputs = tf.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)

	outputs = tf.layers.Flatten()(outputs)

	outputs = tf.layers.Dense(256, activation = "relu")(outputs)
	outputs = tf.layers.Dense(64, activation = "relu")(outputs)
	outputs = tf.layers.Dense(num_classes, activation = "relu")(outputs)

	return outputs

def Simple_VGG_19(inputs, num_classes):

	outputs = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(inputs)
	outputs = Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	outputs = MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)

	outputs = Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	outputs = Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	outputs = MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)

	outputs = Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	outputs = Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	outputs = MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)

	outputs = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	outputs = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	outputs = MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)

	#outputs = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = Conv2D(filters = 512, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)

	outputs = tf.layers.Flatten()(outputs)

	#outputs = tf.layers.Dense(4096, activation = "relu")(outputs)
	#outputs = tf.layers.Dense(4096, activation = "relu")(outputs)
	#outputs = tf.layers.Dense(1000, activation = "relu")(outputs)
	outputs = tf.layers.Dense(512, activation = "relu")(outputs)
	outputs = tf.layers.Dense(512, activation = "relu")(outputs)
	outputs = tf.layers.Dense(64, activation = "relu")(outputs)
	outputs = tf.layers.Dense(num_classes, activation = "relu")(outputs)

	return outputs


def loss(logits, labels):
	#损失函数
	softmax = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
	return tf.reduce_mean(softmax)

def train(loss, lr):
	#训练句柄
	global_step = tf.Variable(0, trainable = False)
	return tf.train.AdamOptimizer(lr).minimize(loss, global_step = global_step)

def evaluation(logits, labels):
	#评价结果
	return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))