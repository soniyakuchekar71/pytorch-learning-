import torch
from torch import nn
from torchvision import models


# --------------------------------------------------
# 1. Load a pretrained ResNet18 model
# --------------------------------------------------

model = models.resnet18(weights="DEFAULT")


# --------------------------------------------------
# 2. Freeze the pretrained layers
# --------------------------------------------------

for parameter in model.parameters():
    parameter.requires_grad = False


# --------------------------------------------------
# 3. Replace the final classification layer
# --------------------------------------------------

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# --------------------------------------------------
# 4. Define the loss function
# --------------------------------------------------

loss_function = nn.CrossEntropyLoss()


# --------------------------------------------------
# 5. Define the optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 6. Create sample image data
# --------------------------------------------------

images = torch.randn(
    4,
    3,
    224,
    224
)

labels = torch.tensor([
    0,
    1,
    0,
    1
])


# --------------------------------------------------
# 7. Forward pass
# --------------------------------------------------

outputs = model(images)


# --------------------------------------------------
# 8. Calculate loss
# --------------------------------------------------

loss = loss_function(
    outputs,
    labels
)


# --------------------------------------------------
# 9. Backward pass
# --------------------------------------------------

optimizer.zero_grad()

loss.backward()

optimizer.step()


# --------------------------------------------------
# 10. Get predictions
# --------------------------------------------------

predictions = outputs.argmax(dim=1)


# --------------------------------------------------
# 11. Display results
# --------------------------------------------------

print(model)

print("\nInput shape:", images.shape)

print("Output shape:", outputs.shape)

print("Labels:", labels)

print("Predictions:", predictions)

print("Loss:", loss.item())
