import torch
import torch.nn as nn


# Create input tensor
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])


# Create activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()


# Apply activation functions
relu_output = relu(x)
sigmoid_output = sigmoid(x)
tanh_output = tanh(x)


# Print results
print("Input:", x)
print("ReLU:", relu_output)
print("Sigmoid:", sigmoid_output)
print("Tanh:", tanh_output)
