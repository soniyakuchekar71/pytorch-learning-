import torch
from torch import nn


class SimpleNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)

        x = self.layer2(x)
        x = torch.relu(x)

        x = self.output(x)

        return x


model = SimpleNeuralNetwork()

x = torch.tensor([[1.0, 2.0]])

prediction = model(x)

print("Model:")
print(model)

print("\nInput:")
print(x)

print("\nPrediction:")
print(prediction)

print("\nModel Parameters:")
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
