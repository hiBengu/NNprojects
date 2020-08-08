import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(988)

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

class trainTest():
    def __init__(self, epoch = 20, colNum = 3):
        self.net = Net(colNum)
        self.epoch = epoch
        self.colNum = colNum
        self.allAverages = []

        self.lossFunction = nn.MSELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def loadDataFromPickle(self):
        with open('xDataTrain.pkl', 'rb') as f:
            self.xData = pickle.load(f)
        with open('yDataTrain.pkl', 'rb') as f:
            self.yData = pickle.load(f)

        with open('xDataTest.pkl', 'rb') as f:
            self.xDataTest = pickle.load(f)
        with open('yDataTest.pkl', 'rb') as f:
            self.yDataTest = pickle.load(f)
        print("Shapes of x and y accordingly:")
        print(self.xData.shape)
        print(self.yData.shape)
        print(type(self.xData))
        print("Data Load is Complete!")
        print("###############")

    def createTrainAndTestData(self, dataSize = 50000):
        self.dataSize = self.xData.shape[0]
        dataSize = self.dataSize

        self.trainX = torch.FloatTensor(self.xData)
        self.trainY = torch.FloatTensor(self.yData)

        self.testX = torch.FloatTensor(self.xDataTest)
        self.testY = torch.FloatTensor(self.yDataTest)

        print("Train and Test data Created as Tensors!")
        print("###############")

    def createDataLoaders(self, batchSize=1000):
        self.batchSize = batchSize
        print(self.trainX.shape)
        print(self.trainY.shape)
        dataset = torch.utils.data.TensorDataset(self.trainX, self.trainY)
        self.trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize)

        testset = torch.utils.data.TensorDataset(self.testX, self.testY)
        self.testLoader = torch.utils.data.DataLoader(testset, batch_size=batchSize)

    def train(self):
        self.testAverages = []
        saveName = 'Model'
        for _ in range(self.epoch):
            self.epochLoss = []
            for x,y in self.trainLoader:
                # y = y-1
                self.net.zero_grad()
                output = self.net(x.view(-1, self.colNum))
                loss = self.lossFunction(output, y.view(-1,10))
                loss.backward()
                self.optimizer.step()
                self.epochLoss.append(loss.item())
            print("Avg. Loss: " + str(sum(self.epochLoss)/len(self.epochLoss)))
            self.allAverages.append(sum(self.epochLoss)/len(self.epochLoss))
            self.testAverages.append(self.evalTest())
        torch.save({'state_dict': self.net.state_dict()}, f"{saveName}.pt")
        with open(f"{saveName}.pkl", 'wb') as f:
            pickle.dump(self.allAverages, f)
        with open(f"{saveName}Test.pkl", 'wb') as f:
            pickle.dump(self.testAverages, f)


    def evalTest(self, last=False):
        difArray = []

        for data in self.testLoader:
            X, y = data
            # y = y-1
            output = self.net(X.view(-1,self.colNum))
            # print(output)
            # print(y)
            # print(output.shape)
            # print(y.shape)
            for i in range(output.shape[0]):
                # print(f"For the input {X},Prediction: {output[i]}, Correct:{y[i]}")
                if last:
                    print(output[i])
                dif = abs(output[i] - y[i])
                difArray.append(dif)
        accArray = sum(difArray)/len(difArray)
        # print("Accuracy: ", accArray)
        trueAcc = np.average(accArray.detach().numpy())
        print("Accuracy: ", trueAcc)
        return trueAcc

train = trainTest(epoch=1000)
train.loadDataFromPickle()
train.createTrainAndTestData()
train.createDataLoaders()
train.train()
train.evalTest(last=True)
