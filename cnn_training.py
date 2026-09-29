import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------
# Lesson 12 — CNN Training and Evaluation
# --------------------------------------------------

torch.manual_seed(42)


# --------------------------------------------------
# 1. Create sample image data
# --------------------------------------------------

X_train = torch.randn(100, 3, 64, 64)
y_train = torch.randint(0, 2, (100,))

X_test = torch.randn(20, 3, 64, 64)
y_test = torch.randint(0, 2, (20,))


# --------------------------------------------------
# 2. Create datasets and dataloaders
# --------------------------------------------------

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False
)


# --------------------------------------------------
# 3. Create CNN model
# --------------------------------------------------

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


model = SimpleCNN()


# --------------------------------------------------
# 4. Loss function and optimizer
# --------------------------------------------------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 5. Training
# --------------------------------------------------

epochs = 5

for epoch in range(epochs):

    # Put model into training mode
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = loss_function(outputs, labels)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Track total loss
        total_loss += loss.item()

        # Get predicted classes
        predictions = outputs.argmax(dim=1)

        # Count correct predictions
        correct += (predictions == labels).sum().item()

        # Count total samples
        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Loss: {train_loss:.4f} | "
        f"Accuracy: {train_accuracy:.2%}"
    )


# --------------------------------------------------
# 6. Evaluation
# --------------------------------------------------

model.eval()

correct = 0
total = 0

# Gradients are not needed during evaluation
with torch.no_grad():

    for images, labels in test_loader:

        # Forward pass
        outputs = model(images)

        # Get predicted classes
        predictions = outputs.argmax(dim=1)

        # Count correct predictions
        correct += (predictions == labels).sum().item()

        # Count total samples
        total += labels.size(0)


# Calculate test accuracy
test_accuracy = correct / total


# --------------------------------------------------
# 7. Display final result
# --------------------------------------------------

print("\nFinal Evaluation")
print(f"Test Accuracy: {test_accuracy:.2%}")
