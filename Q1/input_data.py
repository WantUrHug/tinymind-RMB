import numpy as np
import os
from pprint import pprint
import cv2

def get_file(pic_dir, label_csv, class_dict):

	csv = np.loadtxt(label_csv, dtype = str, skiprows = (1), delimiter= ', ')

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

if __name__ == "__main__":

		train_dir = "D:\\xunleiDownload\\RMB\\train_data\\"
		label_csv = "D:\\xunleiDownload\\RMB\\train_face_value_label.csv"
		
		classes = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"]
		num_classes = len(classes)
		class_dict = {}
		for i, cls in enumerate(classes):
			class_dict[cls] = [1 if j == i else 0 for j in range(10)]

		#pprint(class_dict)

		data_list, label_list = get_file(train_dir, label_csv, class_dict)[:5]
		for i in range(5):
			print(data_list[i], label_list[i])