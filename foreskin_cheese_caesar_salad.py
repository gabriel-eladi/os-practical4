import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class beanNeuralNetwork(nn.Module):
    def __init__(self, trainingDataset: pd.DataFrame, testingDataset: pd.DataFrame, input_size=50, hidden_size=100, output_size=16, learning_rate=0.2, batch_size=50, validation_share=0.2):
        super(beanNeuralNetwork,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.function1 = nn.linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.function2 = nn.linear(hidden_size, output_size)
        self.softMax = nn.Softmax(dim=output_size)

        self.trainingDataset = trainingDataset
        self.testingDataset = testingDataset

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.validation_share = validation_share

        #convert dfs to tensors
        self.trainingTensor = torch.from_numpy(trainingDataset.values)
        self.testingTensor = torch.from_numpy(testingDataset.values)
        self.train_number = len(trainingDataset)
        self.train_indices = list(range(self.train_number))
        np.random.shuffle(self.train_indices) #mix training samples
        self.testing_portion = round(self.train_number * validation_share) #probably 2000 or so
        self.training_index, self.testing_index = self.train_indices[self.testing_portion:], self.train_indices[:self.testing_portion]

    def forward(self, x):
        #self.weight1 = torch.randn(self.input_size, self.hidden_size, requires_grad=True)
        #self.bias1 = torch.randn(1, self.hidden_size, requires_grad=True)
        #self.weight2 = torch.randn(self.hidden_size, self.output_size, requires_grad=True)
        #self.bias2 = torch.randn(1, self.output_size, requires_grad=True)

        x = self.function1(x)
        x = self.relu(x)
        x = self.function2(x)
        x = self.softmax(x)
        return x

