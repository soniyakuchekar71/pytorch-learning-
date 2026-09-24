import torch
import torch.nn as nn


# Create a simple neural network
class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


# Create the model
model = SimpleModel()

# Create the loss function
loss_function = nn.MSELoss()

# Create the optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# Input and target
x = torch.tensor([[2.0]])
target = torch.tensor([[4.0]])


# Training loop
for step in range(10):

    # Clear previous gradients
    optimizer.zero_grad()

    # Forward pass
    prediction = model(x)

    # Calculate loss
    loss = loss_function(prediction, target)

    # Calculate gradients
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Print results
    print(
        "Step:", step + 1,
        "Prediction:", prediction.item(),
        "Loss:", loss.item()
    )
