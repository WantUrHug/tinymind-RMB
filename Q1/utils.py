import matplotlib.pyplot as plt
import os

def show_result(history, steps = 1):
	'''
	在训练时会用一个名为history的字典来储存每一个mini-batch的训练误差、训练准确率、
	验证误差和验证准确率，现在就是利用这个history来画图，从图像中更好的显示训练的过程
	'''

	steps = range(1, len(history["train_loss"])*steps + 1, steps)
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

def save_his_csv(history, filename = None, step = 1, filepath = "D:\\GitFile\\RMB\\Q1\\"):
	'''
	将训练过程中的训练误差、训练精度、验证误差和验证精度都保存成csv，可以在程序结束之后通过csv查看
	'''
	with open(os.path.join(filepath, filename), "w") as f:
		try:
			history["test_loss"]
		except KeyError:
			head = ",".join(["step", "train loss", "train acc"])
			head += "\n"
			#print(head)
			f.writelines(head)
		else:
			head = ",".join(["step", "train loss", "train acc", "test loss", "test acc"])
			head += "\n"
			#print(head)
			f.write(head)
			l = len(history["train_loss"])

			for i in range(l):
				content = ",".join(map(str, [step*(i+1), history["train_loss"][i], history["train_acc"][i], history["test_loss"][i], history["test_acc"][i]]))
				content += "\n"
				#print(content)
				f.write(content)

class csvHelper():

	def __init__(self, path):
		self.path = path

	def writelines(self, char, type = "a+"):
		#默认是续写，type="w"时是重写
		file = open(self.path, type)
		file.writelines(char)
		file.close()

if __name__ == "__main__":

	#简单测试代码可行性
	history = {}
	history["train_loss"] = [1,2,3,4,5]
	history["test_loss"] = [2,3,4,5,6]
	history["train_acc"] = [3,4,5,6,7]
	history["test_acc"] = [4,5,6,7,8]

	save_his_csv(history)