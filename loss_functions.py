import torch
import torch.nn as nn


# Prediction made by the model
prediction = torch.tensor([2.5])


# Actual target value
actual = torch.tensor([3.0])


# Create Mean Squared Error loss
mse_loss = nn.MSELoss()


# Calculate loss
loss = mse_loss(prediction, actual)


print("Prediction:", prediction)
print("Actual:", actual)
print("MSE Loss:", loss)
