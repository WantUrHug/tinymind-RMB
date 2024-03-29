import numpy as np
import os
import shutil
from pprint import pprint
import cv2

IMAGE_H = 224
IMAGE_W = 224

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
	image_h = IMAGE_H,
	image_w = IMAGE_W,
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
	image_h = IMAGE_H,
	image_w = IMAGE_W,
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
			#不可避免可能有一些图片是残缺的，尺寸太小，此时抛出异常并不能解决问题，
			#要把太小的不能随机截取的图片用RESIZE的方法处理
			#raise ValueError("the picture is too small to crop.")
			return cv2.resize(im, (image_h, image_w))/255.0

		start_h = np.random.randint(height - image_h)
		start_w = np.random.randint(width - image_w)
		#也是要变换到0-1
		return im[start_h: start_h + image_h, start_w: start_w + image_w, :]/255.0


def one2batchbyRANDOMCROP(test_dir, batch_size = 10, image_h = IMAGE_H, image_w = IMAGE_W, channels = 3):
	'''
	专为训练数据准备的，在随机裁剪的条件下，把一张照片每次切一个224x224出来，组成一个batch，给
	进去网络里，会出来batch个结果，选择其中投票次数高的作为该图片最终的输出结果
	建议写成生成器的形式，没有必要再用dataset来写，直接用一个for循环即可
	生成的内容为图片名+batch数据
	'''
	for img_name in os.listdir(test_dir):
		data = np.zeros([batch_size, image_h, image_w, channels])
		for i in range(batch_size):
			data[i] = get_pixel(os.path.join(test_dir, img_name), image_h, image_w, method = "RANDOM_CROP")
		yield img_name, data

def batch2batchbyRANDOMCROP(test_dir, img_num = 4, batch_size = 10, image_h = IMAGE_H, image_w = IMAGE_W, channels = 3):
	'''
	专为训练数据准备的，在随机裁剪的条件下，把一张照片每次切一个224x224出来，组成一个batch，给
	进去网络里，会出来batch个结果，选择其中投票次数高的作为该图片最终的输出结果
	建议写成生成器的形式。但是每次只有一张照片，测试的效率太低，使用CPU训练需要10个小时，而即使是
	使用GPU也是要两个半小时左右，需要这个函数相比one2batchbyRANDOMCROP，每次会处理img_num张照片，也就是
	每次喂给训练完成的模型img_num*batch_size个随机截取的图片，提高整体的测试效率。
	'''

	name_list = os.listdir(test_dir)
	total = len(name_list) #20000
	if total%img_num == 0: #20000%4=0
		epcohs = int(total/img_num) #epochs=5000
		for i in range(epcohs):
			img_names = name_list[i*img_num: (i + 1)*img_num]
			data = np.zeros([img_num*batch_size, image_h, image_w, channels])
			for j in range(img_num):
				for k in range(batch_size):
					data[j*batch_size + k] = get_pixel(os.path.join(test_dir, img_names[j]), image_h, image_w, method = "RANDOM_CROP")
			yield img_names, data
	else: # total%img_num != 0
		epcohs = total//img_num
		for i in range(epcohs):
			img_names = name_list[i*img_num, (i + 1)*img_num]
			data = np.zeros([img_num*batch_size, image_h, image_w, channels])
			for j in range(img_num):
				for k in range(batch_size):
					data[j] = get_pixel(os.path.join(test_dir, img_name), image_h, image_w, method = "RANDOM_CROP")
			yield img_names, data
		name_remainder = name_list[epcohs*img_num:]
		size_remainder = len(name_remainder)
		print("Remainder size is %d."%size_remainder)
		data = np.zeros([size_remainder*batch_size, image_h, image_w, channels])
		for i, j in enumerata(name_remainder):
			data[i] = get_pixel(os.path.join(test_dir, img_name), image_h, image_w, method = "RANDOM_CROP")
		yield name_remainder, data
		
if __name__ == "__main__":

		#观察缩放的效果
		train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
		dst_dir = "D:\\xunleiDownload\\RMB\\sp_data"
		test_dir = "D:\\xunleiDownload\\RMB\\public_test_data\\public_test_data"
		for name_list, data in batch2batchbyRANDOMCROP(test_dir):
			print(data[1])