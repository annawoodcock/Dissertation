import torch
import torch.nn.functional as F

class NeuralNet_Relu1(torch.nn.Module):
    def __init__(self,input_size=21, hidden=10, output=1):
        super(NeuralNet_Relu1, self).__init__()        
        self.fc1 = torch.nn.Linear(input_size, hidden)
        self.fc2 = torch.nn.Linear(hidden, output)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x