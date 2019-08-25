import matplotlib.pyplot as plt

def show_result(history):
	'''
	在训练时会用一个名为history的字典来储存每一个mini-batch的训练误差、训练准确率、
	验证误差和验证准确率，现在就是利用这个history来画图，从图像中更好的显示训练的过程
	'''

	steps = range(len(history["train_loss"]))
	test_flag = True

	#检查一下其中是否有验证的结果，来决定我们后续的画图.入股有验证的结果，那就
	#把训练和验证的误差放在同一张图，把训练和验证的准确率放在同一张图;否则就是
	#把训练的误差和准确率放在同一张图
	try:
		history["test_loss"]
	except KeyError:
		print("No testing history, only training.")
		test_flag = False

	if not test_flag:
		plt.plot(steps, history["train_loss"], "b", label = "train loss")
		plt.plot(steps, history["train_acc"], "r", label = "train accuracy")
	else:
		plt.subplot(1,2,1)
		plt.plot(steps, history["train_loss"], "b", label = "train loss")
		plt.plot(steps, history["test_loss"], "r", label = "test loss")

		plt.subplot(1,2,2)
		plt.plot(steps, history["train_acc"], "b", label = "train accuracy")
		plt.plot(steps, history["test_acc"], "r", label = "test accuracy")

	plt.legend()
	plt.show()