import numpy as np
import os
import shutil
from pprint import pprint
import cv2

def split_train_test(src, dst, test_size = 0.1):

	train_dir = os.path.join(dst, "train")
	validation_dir = os.path.join(dst, "test")
	amount = len(os.listdir(src))
	train_amount = int(amount*(1-test_size))

	if not os.path.exists(train_dir):
		os.mkdir(train_dir)
		for i in os.listdir(src)[:train_amount]:
			shutil.copy(os.path.join(src, i), train_dir)
		print("Finish to copy train data.")
	else:
		print("You have set up train dir to contain train data.")

	if not os.path.exists(validation_dir):
		os.mkdir(validation_dir)
		for i in os.listdir(src)[train_amount:]:
			shutil.copy(os.path.join(src, i), validation_dir)
		print("Finish to copy test data.")
	else:
		print("You have set up test dir to contain test data.")

def get_file(pic_dir, label_csv):

	classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
	NUM_CLASSES = len(classes)
	class_dict = {}
	for i, cls in enumerate(classes):
		class_dict[cls] = [1 if j == i else 0 for j in range(NUM_CLASSES)]

	csv = np.loadtxt(label_csv, dtype = str, skiprows = (1), delimiter= ', ')

	if test_num >= len(csv):
		raise ValueError("the parameter test_num is to large so that there are no training data.")

	csv_dict = {}

	for i in range(len(csv)):
		csv_dict[csv[i, 0]] = csv[i, 1]

	#pprint(csv_dict)
	data = []
	labels = []

	for i in os.listdir(pic_dir):
		data.append(os.path.join(pic_dir, i))
		label = class_dict[csv_dict[i]]
		labels.append(label)

	return data, labels

def args_wrapper(*args, **kwargs):
	#Forbid others to pass parameters when using.
	def outer_wrapper(fun):
		def inner_wrapper():
			return fun(*args, **kwargs)
		return inner_wrapper
	return outer_wrapper

@args_wrapper(
	dir = "D:\\xunleiDownload\\RMB\\sp_data\\train",
	label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv",
	image_h = 224,
	image_w = 224)
def train_data_gen(dir, label_csv, image_h, image_w):
	
	classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
	NUM_CLASSES = len(classes)
	class_dict = {}
	for i, cls in enumerate(classes):
		class_dict[cls] = [1 if j == i else 0 for j in range(NUM_CLASSES)]

	csv = np.loadtxt(label_csv, dtype = str, skiprows = (1), delimiter= ', ')

	#turn np.ndarray to dict.
	csv_dict = {}
	for i in range(len(csv)):
		csv_dict[csv[i, 0]] = csv[i, 1]

	for i in os.listdir(dir):

		yield get_pixel(os.path.join(dir, i), image_h, image_w), class_dict[csv_dict[i]]

@args_wrapper(
	dir = "D:\\xunleiDownload\\RMB\\sp_data\\test",
	label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv",
	image_h = 224,
	image_w = 224)
def test_data_gen(dir, label_csv, image_h, image_w):
	
	classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
	NUM_CLASSES = len(classes)
	class_dict = {}
	for i, cls in enumerate(classes):
		class_dict[cls] = [1 if j == i else 0 for j in range(NUM_CLASSES)]

	csv = np.loadtxt(label_csv, dtype = str, skiprows = (1), delimiter= ', ')

	#turn np.ndarray to dict.
	csv_dict = {}
	for i in range(len(csv)):
		csv_dict[csv[i, 0]] = csv[i, 1]

	for i in os.listdir(dir):

		yield get_pixel(os.path.join(dir, i), image_h, image_w), class_dict[csv_dict[i]]

def get_pixel(src, image_h, image_w, method = "RESIZE"):

	if method == "RESIZE":
		im = cv2.imread(src)
		im = cv2.resize(im, (image_h, image_w))

		return im/255.0

	elif method == "RANDOM_CUT":
		raise ValueError("Have not set up this method!")










if __name__ == "__main__":

		train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
		dst_dir = "D:\\xunleiDownload\\RMB\\sp_data"
		split_train_test(train_dir, dst_dir)