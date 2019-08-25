import numpy as np
import os
import shutil
from pprint import pprint
import cv2
from tensorflow import random_crop

def split_train_test(src, dst, test_size = 0.1):
	'''
	原始的数据是所有图片都在一个文件夹，所以要分成一部分做训练和做检验，这种划分方式有助于后面根据文件
	夹创建各自的数据生成器.
	src是图片所在的路径
	dst是分开之后图片各自在的路径
	'''

	#创建两个新文件夹来存放训练数据和检验数据
	train_dir = os.path.join(dst, "train")
	validation_dir = os.path.join(dst, "test")

	#计算一共有多少张图片，根据test_size的比例来划分
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

def args_wrapper(*args, **kwargs):
	#定义一个函数，生成的结果是一个装饰器，作用是换个形式填充默认值，同时
	#被装饰的函数就无法再传入参数
	def outer_wrapper(fun):
		def inner_wrapper():
			return fun(*args, **kwargs)
		return inner_wrapper
	return outer_wrapper

@args_wrapper(
	dir = "D:\\xunleiDownload\\RMB\\sp_data\\train",
	label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv",
	image_h = 224,
	image_w = 224,
	method = "RANDOM_CROP")
def train_data_gen(dir, label_csv, image_h, image_w, method):
	
	#一共有10类，创建一个字典来把面值转化成稀疏向量表示
	classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
	NUM_CLASSES = len(classes)
	class_dict = {}
	for i, cls in enumerate(classes):
		class_dict[cls] = [1 if j == i else 0 for j in range(NUM_CLASSES)]

	#然后要解析这个csv文件，把里面的序号和面值的关系转化成字典，来支持我们的索引
	csv = np.loadtxt(label_csv, dtype = str, skiprows = (1), delimiter= ', ')
	csv_dict = {}
	for i in range(len(csv)):
		csv_dict[csv[i, 0]] = csv[i, 1]

	#一个一个的yield数据结果
	for i in os.listdir(dir):

		yield get_pixel(os.path.join(dir, i), image_h, image_w, method), class_dict[csv_dict[i]]

@args_wrapper(
	dir = "D:\\xunleiDownload\\RMB\\sp_data\\test",
	label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv",
	image_h = 224,
	image_w = 224,
	method = "RANDOM_CROP")
def test_data_gen(dir, label_csv, image_h, image_w, method):
	
	#同train_data_gen.
	classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
	NUM_CLASSES = len(classes)
	class_dict = {}
	for i, cls in enumerate(classes):
		class_dict[cls] = [1 if j == i else 0 for j in range(NUM_CLASSES)]

	csv = np.loadtxt(label_csv, dtype = str, skiprows = (1), delimiter= ', ')
	csv_dict = {}
	for i in range(len(csv)):
		csv_dict[csv[i, 0]] = csv[i, 1]

	for i in os.listdir(dir):

		yield get_pixel(os.path.join(dir, i), image_h, image_w, method), class_dict[csv_dict[i]]



def get_pixel(src, image_h, image_w, method = "RESIZE"):
	'''
	将一张图片处理成可以传递给网络的尺寸，通常是RESIZE，但是也可以使用其他的方法，
	例如RANDOM_CROP随机裁剪，也有更好的作用，一来避免了RESIZE的计算开销，二来避免了
	比例缩放时对物体的改变.
	src是文件的路径.
	'''

	if method == "RESIZE":
		
		im = cv2.imread(src)
		#使用opencv的缩放方法
		im = cv2.resize(im, (image_h, image_w))
		#切记要把像素值从0-255的uint8缩放到0-1之间的浮点数
		return im/255.0

	elif method == "RANDOM_CROP":
		
		im = cv2.imread(src)
		#原先想使用tf.random_crop方法，但是返回的是张量，算了
		#自己来写方法
		height, width, _ = im.shape

		if height < image_h or width < image_w:
			raise ValueError("the picture is too small to crop.")

		start_h = np.random.randint(height - image_h)
		start_w = np.random.randint(width - image_w)
		#也是要变换到0-1
		return im[start_h: start_h + image_h, start_w: start_w + image_w, :]/255.0






if __name__ == "__main__":

		#观察缩放的效果
		train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
		dst_dir = "D:\\xunleiDownload\\RMB\\sp_data"
		test_pic = "D:\\xunleiDownload\\RMB\\sp_data\\train\\0A2PDULI.jpg"
		new = get_pixel(test_pic, 224, 224, "RANDOM_CROP")
		cv2.imshow("imshow", new)
		cv2.waitKey(0)
		cv2.destroyAllWindows()