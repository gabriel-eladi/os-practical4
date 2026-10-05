import torch
import torch.nn as nn

class beanNeuralNetwork(nn.Module):
    def __init__(self, input_size=50, hidden_size=100, output_size=16):
        super(beanNeuralNetwork,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.function1 = nn.linear(input_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)
        self.function2 = nn.linear(hidden_size, output_size)
        self.softMax = nn.Softmax(dim=output_size)

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

