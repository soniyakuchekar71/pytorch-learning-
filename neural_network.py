import torch
import torch.nn as nn


# Create a simple neural network
class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        # 2 input features and 1 output
        self.layer = nn.Linear(2, 1)

    # Define how data moves through the model
    def forward(self, x):
        return self.layer(x)


# Create the model
model = SimpleModel()

# Create input data
x = torch.tensor([[2.0, 3.0]])

# Pass the input through the model
output = model(x)

# Print the results
print("Input:", x)
print("Output:", output)

# Display the model structure
print("\nModel:")
print(model)

# Display the layer weights
print("\nWeights:")
print(model.layer.weight)

# Display the bias
print("\nBias:")
print(model.layer.bias)
