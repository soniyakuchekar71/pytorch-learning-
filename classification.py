import torch
from torch import nn


# Training data
# Each row contains two features.
X = torch.tensor([
    [1.0, 1.0],
    [1.5, 1.2],
    [2.0, 1.8],
    [2.5, 2.2],
    [3.0, 3.0],
    [3.5, 3.2],
    [4.0, 4.0],
    [4.5, 4.2],
], dtype=torch.float32)

# Binary labels:
# 0 = Class 0
# 1 = Class 1
y = torch.tensor([
    [0.0],
    [0.0],
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
    [1.0],
], dtype=torch.float32)


# Neural network for binary classification
class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output(x)
        return x


model = Classifier()

# Loss function for binary classification
loss_function = nn.BCEWithLogitsLoss()

# Adam optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# Training loop
epochs = 1000

for epoch in range(epochs):
    # Forward pass
    predictions = model(X)

    # Calculate loss
    loss = loss_function(predictions, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Backward pass
    loss.backward()

    # Update model parameters
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


# Make predictions
with torch.no_grad():
    logits = model(X)

    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Convert probabilities into class predictions
    predicted_classes = (probabilities >= 0.5).float()


print("\nProbabilities:")
print(probabilities)

print("\nPredicted Classes:")
print(predicted_classes)

print("\nActual Classes:")
print(y)
