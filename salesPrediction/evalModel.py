from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

class Net(nn.Module):
    def __init__(self, nInput):
        super().__init__()
        self.fc1 = nn.Linear(nInput,64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

# Load Model
net = Net(3)
model = torch.load(f"kukumavModel.pt", map_location=torch.device('cpu')) # Load Model
net.load_state_dict(model['state_dict'])

# Load data from pickle to dataloader
with open('xDataTest.pkl', 'rb') as f:
    xDataTest = pickle.load(f)
with open('yDataTest.pkl', 'rb') as f:
    yDataTest = pickle.load(f)

testX = torch.FloatTensor(xDataTest)
testY = torch.FloatTensor(yDataTest)

testset = torch.utils.data.TensorDataset(testX, testY)
testLoader = torch.utils.data.DataLoader(testset, batch_size=1)

# Calculate the error in trainTest
difArray = []
predArray = []
realArray = []

for data in testLoader:
    X, y = data
    # y = y-1
    output = net(X.view(-1,3))
    for i in range(output.shape[0]):
        # print(output[i], y[i])
        dif = abs(output[i] - y[i])
        print(y[i])
        predArray.append(output[i].detach().numpy())
        realArray.append(y[i].detach().numpy())
        difArray.append(dif)

print("Accuracy: ", sum(difArray)/len(difArray))

graphPred = []
graphReal = []

for i in range(10):
    for day in predArray:
        graphPred.append(day[i])

    for day in realArray:
        graphReal.append(day[i])

print(graphPred)
print("bitti")
print(graphReal)

for i in range(10):
    print(i)
    plt.xlabel('Gunler', fontsize=18)
    plt.ylabel('Tahmini ve Gercek Satis Degerleri', fontsize=16)
    plt.plot(graphPred[i*26:(i+1)*26], label="Tahmini Satis Degerleri") # plotting by columns
    plt.plot(graphReal[i*26:(i+1)*26], label="Gercek Satis Degerleri") # plotting by columns

    plt.legend()
    plt.show()
