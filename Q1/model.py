import tensorflow as tf

#build model
def inference(inputs, num_classes):

	#inputs.shape = [None, IMAGE_H, IMAGE_W, CHANNEL]
	outputs = tf.layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(inputs)
	#shape = [None, IMAGE_H, IMAGE_W, 16]
	#outputs = tf.layers.Conv2D(filters = 32, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#shape = [None, IMAGE_H, IMAGE_W, 32]
	outputs = tf.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)
	#shape = [None, IMAGE_H/2, IMAGE_W/2, 32]
	outputs = tf.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#shape = [None, IMAGE_H/2, IMAGE_W/2, 64]
	outputs = tf.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)
	#shape = [None, IMAGE_H/4, IMAGE_W/4, 128]
	#outputs = tf.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = "relu", padding = "SAME")(outputs)
	#outputs = tf.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2))(outputs)
	outputs = tf.layers.Flatten()(outputs)
	#shape = [None, IAMGE_H*IMAGE_W*8]
	outputs = tf.layers.Dense(512, activation = "relu")(outputs)
	outputs = tf.layers.Dense(256, activation = "relu")(outputs)
	outputs = tf.layers.Dense(num_classes, activation = "relu")(outputs)

	return outputs

def loss(logits, labels):

	softmax = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
	return tf.reduce_mean(softmax)

def train(loss, lr):
	return tf.train.GradientDescentOptimizer(lr).minimize(loss)

def evaluation(logits, labels):
	return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), tf.float32))