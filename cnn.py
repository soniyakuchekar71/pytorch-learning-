import torch
from torch import nn


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# Create the model
model = SimpleCNN()

# Create a sample image
# Shape: [batch_size, channels, height, width]
x = torch.randn(1, 3, 64, 64)

# Pass the image through the model
output = model(x)

print("Model:")
print(model)

print("\nInput shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)
