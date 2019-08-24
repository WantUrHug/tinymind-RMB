import matplotlib.pyplot as plt

def show_result(history):

	steps = range(len(history["train_loss"]))
	test_flag = True

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