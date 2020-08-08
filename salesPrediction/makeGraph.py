from matplotlib import pyplot as plt
import pickle
import numpy as np

file = "Model.pkl"

with open(file, 'rb') as f:
    data = pickle.load(f, encoding='bytes')

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss and Accuracy', fontsize=16)
plt.plot(data, label="Train Dataset") # plotting by columns

file = "ModelTest.pkl"

with open(file, 'rb') as f:
    data = pickle.load(f, encoding='bytes')

# print(f"Minimum if this Array: {np.amin(data)} and it's index: {np.where(data == np.amin(data))}")

plt.plot(data, label="Test Dataset") # plotting by columns

plt.legend()
plt.show()
