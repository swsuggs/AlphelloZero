import numpy as np
import cPickle as cpkl
from matplotlib import pyplot as plt

# loss = np.array(cpkl.load(open('loss.cpkl','rb'))).flatten()
# print(loss)
# plt.plot(range(1, len(loss)+1), loss)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Training Loss")
# plt.savefig("loss_plot.png")
# # plt.show()


loss = np.array(cpkl.load(open('first_attempt_loss.cpkl','rb'))).flatten()
print(loss)
plt.plot(range(1, len(loss)+1), loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("first_loss_plot.png")
plt.show()
