# PyTorch Learning

This repository contains my notes, practice code, and experiments while learning PyTorch from the basics.

The focus is on understanding core concepts step by step and applying them through practical examples.

---

## Lesson 1 — Tensors

**File:** `tensors.py`

### Objectives

* Understand what tensors are and why they are used in PyTorch.
* Learn how to create tensors.
* Understand tensor dimensions, shapes, and data types.
* Perform basic tensor operations.
* Learn how to reshape tensors.

### Key Concepts

* `torch.tensor()`
* `torch.zeros()`
* `torch.ones()`
* `torch.rand()`
* Tensor shape
* Tensor dimensions
* Data types
* Reshaping with `.reshape()`

### What I Learned

Tensors are the basic data structure used by PyTorch. They are similar to arrays but are designed for efficient numerical computation and can also be used with GPUs.

---

## Lesson 2 — Autograd

**File:** `autograd.py`

### Objectives

* Understand automatic differentiation.
* Learn how `requires_grad` works.
* Understand `.backward()`.
* Access gradients using `.grad`.

### Key Concepts

* `requires_grad=True`
* Computational graph
* `.backward()`
* `.grad`

### What I Learned

PyTorch automatically tracks operations on tensors that require gradients. Calling `.backward()` calculates the gradients needed during model training.

---

## Lesson 3 — Neural Network Basics

**File:** `neural_network.py`

### Objectives

* Understand the basic structure of a neural network.
* Learn about inputs, weights, biases, and outputs.
* Understand `nn.Module`.
* Create a simple neural network.

### Key Concepts

* `torch.nn`
* `nn.Module`
* `nn.Linear`
* `forward()`
* Weights
* Biases

### What I Learned

A neural network consists of layers that transform input data into predictions. In PyTorch, neural networks are commonly created by inheriting from `nn.Module`.

---

## Lesson 4 — Activation Functions

**File:** `activation_functions.py`

### Objectives

* Understand why activation functions are needed.
* Learn commonly used activation functions.
* Apply activation functions to neural network outputs.

### Key Concepts

* ReLU
* Sigmoid
* Tanh
* Non-linearity
* `torch.relu()`
* `nn.ReLU()`

### What I Learned

Activation functions introduce non-linearity into neural networks, allowing them to learn more complex patterns.

---

## Lesson 5 — Loss Functions

**File:** `loss_functions.py`

### Objectives

* Understand the purpose of a loss function.
* Learn how predictions are compared with actual values.
* Understand how loss is used during training.

### Key Concepts

* Loss
* Prediction
* Target
* `nn.MSELoss()`
* `nn.CrossEntropyLoss()`

### What I Learned

A loss function measures how different the model's prediction is from the expected output. The model uses this information to improve its parameters.

---

## Lesson 6 — Training a Neural Network

**File:** `training.py`

### Objectives

* Understand the basic training process.
* Learn the training loop.
* Use forward propagation.
* Calculate loss.
* Calculate gradients.
* Update model parameters.

### Training Process

```text
Input
  ↓
Model
  ↓
Prediction
  ↓
Loss
  ↓
Backward Pass
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated Parameters
```

### Key Concepts

* Forward pass
* Loss calculation
* `loss.backward()`
* Optimizer
* `optimizer.step()`
* `optimizer.zero_grad()`
* Training loop

### What I Learned

Training a neural network involves repeatedly making predictions, calculating the loss, computing gradients, and updating the model parameters.

---

## Lesson 7 — Optimizers

**File:** `optimizers.py`

### Objectives

* Understand what an optimizer does.
* Learn how optimizers update model parameters.
* Understand learning rate.
* Compare basic optimizers.

### Key Concepts

* `torch.optim`
* SGD
* Adam
* Learning rate
* Parameter updates

### Example

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

### What I Learned

Optimizers use the gradients calculated during backpropagation to update the model parameters. The learning rate controls how large these updates are.

---

## Lesson 8 — Dataset and DataLoader

**File:** `dataloader.py`

### Objectives

* Understand how datasets are represented in PyTorch.
* Learn how to use `Dataset`.
* Learn how `DataLoader` works.
* Understand batches.
* Shuffle training data.
* Prepare data for model training.

### Key Concepts

* `Dataset`
* `DataLoader`
* Batching
* Shuffling
* `__len__()`
* `__getitem__()`
* `batch_size`

### Example

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


dataset = MyDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)
```

### What I Learned

A `Dataset` stores and provides access to training data, while a `DataLoader` makes it easier to load the data in batches during training.

Using batches is important because models usually train on groups of samples instead of processing the entire dataset at once.

---

## Lesson 9 — Building Neural Networks with `nn.Module`

**File:** `model.py`

### Objectives

* Understand how to build a custom neural network.
* Learn how `nn.Module` is used to create models.
* Create multiple neural network layers.
* Understand the `forward()` method.
* Understand how data flows through different layers.
* Access model parameters.

### Key Concepts

* `nn.Module`
* `nn.Linear`
* `forward()`
* Model architecture
* Layers
* Weights and biases
* Model parameters

### Example

```python
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

print(model)
print("Input:", x)
print("Prediction:", prediction)
```

### Model Architecture

```text
Input: 2 features
       ↓
Linear(2 → 8)
       ↓
ReLU
       ↓
Linear(8 → 4)
       ↓
ReLU
       ↓
Linear(4 → 1)
       ↓
Output
```

### What I Learned

`nn.Module` is the base class used to create neural network models in PyTorch. Layers such as `nn.Linear` can be defined inside the model, while the `forward()` method defines how input data moves through the network.

PyTorch automatically tracks the weights and biases of the layers as model parameters, which can later be updated by an optimizer during training.

---

## Lesson 10 — Classification with PyTorch

**File:** `classification.py`

### Objectives

* Understand what classification means in machine learning.
* Learn the difference between binary classification and regression.
* Build a neural network for binary classification.
* Understand class labels.
* Learn how `BCEWithLogitsLoss()` is used for binary classification.
* Convert model outputs into probabilities.
* Convert probabilities into predicted classes.

### Classification

Classification is a machine learning task where a model predicts a category or class.

For example:

```text
Input Features
      ↓
Neural Network
      ↓
Prediction
      ↓
Class
```

In binary classification, there are two possible classes:

```text
0 → Class 0
1 → Class 1
```

For example, a model could classify an input as:

```text
0 → Not detected
1 → Detected
```

### Key Concepts

* Binary classification
* Class labels
* Logits
* Probability
* Threshold
* `torch.sigmoid()`
* `nn.BCEWithLogitsLoss()`

### Important Concept — Logits

The final layer of the model produces a raw value called a **logit**.

The logit is not directly a probability.

We can convert the logit into a probability using the sigmoid function:

```python
probability = torch.sigmoid(logit)
```

The output is between `0` and `1`.

For example:

```text
Logit → Probability

  2.0 → 0.88
  0.0 → 0.50
 -2.0 → 0.12
```

A common threshold for binary classification is `0.5`:

```python
predicted_class = (probability >= 0.5).float()
```

This means:

```text
Probability >= 0.5 → Class 1
Probability < 0.5  → Class 0
```

### Loss Function

For binary classification, this lesson uses:

```python
loss_function = nn.BCEWithLogitsLoss()
```

`BCEWithLogitsLoss()` combines the sigmoid operation and binary cross-entropy loss in a numerically stable way.

Because of this, the model should return the raw logits during training rather than applying `torch.sigmoid()` inside the model.

### Model Architecture

```text
Input: 2 features
       ↓
Linear(2 → 8)
       ↓
ReLU
       ↓
Linear(8 → 4)
       ↓
ReLU
       ↓
Linear(4 → 1)
       ↓
Logit
       ↓
Sigmoid
       ↓
Probability
       ↓
Threshold 0.5
       ↓
Class 0 or 1
```

### Training Process

```text
Input Data
    ↓
Model
    ↓
Logits
    ↓
BCEWithLogitsLoss
    ↓
Backward Pass
    ↓
Gradients
    ↓
Adam Optimizer
    ↓
Updated Parameters
```

### What I Learned

Classification is used when the output represents a category instead of a continuous value.

In binary classification, the model produces one logit. `BCEWithLogitsLoss()` compares this logit with the target class and calculates the loss.

After training, `torch.sigmoid()` can be used to convert logits into probabilities, and a threshold such as `0.5` can be used to obtain the final class prediction.

---

## Lesson 11 — CNNs and Image Data

**File:** `cnn.py`

### Objectives

* Understand what Convolutional Neural Networks (CNNs) are.
* Learn why CNNs are useful for image data.
* Understand convolutional layers.
* Learn about pooling layers.
* Understand feature maps.
* Build a simple CNN using PyTorch.
* Understand the basic flow of image data through a CNN.

### Why CNNs?

Traditional fully connected neural networks can process image data, but they become inefficient as image size increases.

CNNs are designed to work with images by learning spatial patterns such as:

```text
Edges
  ↓
Shapes
  ↓
Textures
  ↓
More complex features
  ↓
Object / Class
```

CNNs are commonly used for tasks such as:

* Image classification
* Object detection
* Medical image analysis
* Face recognition
* Image segmentation

### Image Representation

Images are represented as tensors.

A color image usually has three channels:

```text
RGB

Red
Green
Blue
```

A single image can have the shape:

```text
[Channels, Height, Width]
```

For example:

```text
[3, 64, 64]
```

When multiple images are processed together in a batch:

```text
[Batch Size, Channels, Height, Width]
```

For example:

```text
[32, 3, 64, 64]
```

This means:

```text
32    → number of images
3     → RGB channels
64    → image height
64    → image width
```

### Convolution

A convolutional layer uses small filters, also called kernels, to scan across an image.

The filters learn useful patterns from the input image.

For example:

```text
Input Image
     ↓
Convolution
     ↓
Feature Maps
```

Early layers may learn simple features such as:

```text
Edges
Lines
Corners
```

Deeper layers can learn more complex patterns.

### Key Concepts

* CNN
* Convolution
* Kernel
* Feature map
* Channels
* `nn.Conv2d`
* Pooling
* `nn.MaxPool2d`
* Flattening
* `nn.Flatten()`

### Important PyTorch Layers

A convolutional layer can be created using:

```python
nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3
)
```

Here:

```text
in_channels = 3
```

means the input image has three channels.

```text
out_channels = 16
```

means the layer learns 16 filters and produces 16 feature maps.

```text
kernel_size = 3
```

means the convolution uses a 3 × 3 kernel.

### Pooling

Pooling reduces the spatial size of feature maps.

A commonly used pooling layer is:

```python
nn.MaxPool2d(kernel_size=2)
```

It reduces the height and width of the feature map while keeping important information.

For example:

```text
Before:

32 × 32

      ↓ MaxPool2d(2)

After:

16 × 16
```

### Simple CNN Architecture

```text
Input Image
[3 × 64 × 64]
      ↓
Conv2d
      ↓
ReLU
      ↓
MaxPool
      ↓
Conv2d
      ↓
ReLU
      ↓
MaxPool
      ↓
Flatten
      ↓
Linear
      ↓
Output
```

### Example

```python
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


model = SimpleCNN()

x = torch.randn(1, 3, 64, 64)

output = model(x)

print(model)
print("Input shape:", x.shape)
print("Output shape:", output.shape)
```

### Understanding the Shape Changes

The input is:

```text
[1, 3, 64, 64]
```

The first convolution changes the channels:

```text
[1, 16, 62, 62]
```

After max pooling:

```text
[1, 16, 31, 31]
```

The second convolution produces:

```text
[1, 32, 29, 29]
```

After the second max pooling:

```text
[1, 32, 14, 14]
```

The tensor is then flattened:

```text
32 × 14 × 14
```

which becomes:

```text
6272
```

The final linear layer produces:

```text
2 outputs
```

representing two possible classes.

### CNN Training Flow

```text
Image
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
  ↓
Convolution
  ↓
ReLU
  ↓
Pooling
  ↓
Flatten
  ↓
Linear Layer
  ↓
Prediction
  ↓
Loss
  ↓
Backward Pass
  ↓
Optimizer
```

### What I Learned

CNNs are neural networks designed to process data with spatial structure, especially images.

Convolutional layers learn visual features from images, while pooling layers reduce the spatial dimensions of feature maps.

I also learned that image tensors in PyTorch usually follow the format:

```text
[Batch Size, Channels, Height, Width]
```

Understanding tensor shapes is important when building CNN architectures because each layer expects a specific input shape.

---

Lesson 12 — CNN Training and Evaluation

File: cnn_training.py

Objectives
Learn how to train a CNN using image data.
Understand the complete CNN training loop.
Use CrossEntropyLoss() for multi-class classification.
Train a model using an optimizer.
Understand training loss and accuracy.
Learn the difference between training and evaluation mode.
Use model.train() and model.eval().
Evaluate a trained CNN without calculating gradients.
Understand how to calculate classification accuracy.
Key Concepts
CNN training
Training loop
CrossEntropyLoss()
model.train()
model.eval()
torch.no_grad()
Accuracy
Batch training
Forward pass
Backward pass
Optimizer
Evaluation
Training vs Evaluation

During training, the model learns by updating its parameters:

Training Data
     ↓
CNN
     ↓
Predictions
     ↓
Loss
     ↓
Backward Pass
     ↓
Gradients
     ↓
Optimizer
     ↓
Updated Parameters

During evaluation, the model only makes predictions:

Test Data
    ↓
CNN
    ↓
Predictions
    ↓
Compare with Labels
    ↓
Accuracy

No parameter updates happen during evaluation.

model.train()

Before training:

model.train()

This tells PyTorch that the model is in training mode.

This is particularly important for layers such as:

Dropout
Batch Normalization
model.eval()

Before evaluation:

model.eval()

This switches the model into evaluation mode.

torch.no_grad()

During evaluation, gradients are not required:

with torch.no_grad():
    outputs = model(images)

This reduces unnecessary computation and memory usage.

Loss Function

For a CNN performing multi-class classification:

loss_function = nn.CrossEntropyLoss()

For example, if there are two classes:

Class 0
Class 1

the model produces two output values for each image.

The predicted class can be obtained using:

predictions = outputs.argmax(dim=1)
Accuracy

Accuracy tells us how many predictions were correct.

correct = (predictions == labels).sum().item()

accuracy = correct / total

For example:

Correct predictions = 18
Total images = 20

Accuracy = 18 / 20
         = 90%
Example
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
# 3. Create CNN
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

        # Update parameters
        optimizer.step()

        # Track loss
        total_loss += loss.item()

        # Get predicted classes
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
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

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)


test_accuracy = correct / total

print("\nTest Accuracy:", f"{test_accuracy:.2%}")
Complete Training Flow
Image Dataset
      ↓
Dataset
      ↓
DataLoader
      ↓
CNN
      ↓
Predictions
      ↓
CrossEntropyLoss
      ↓
optimizer.zero_grad()
      ↓
loss.backward()
      ↓
optimizer.step()
      ↓
Updated CNN
      ↓
Evaluation
      ↓
Accuracy
What I Learned

A CNN needs both training and evaluation phases.

During training, the model performs a forward pass, calculates the loss, computes gradients using backpropagation, and updates its parameters using an optimizer.

During evaluation, the model switches to evaluation mode using model.eval() and predictions are made inside torch.no_grad() because the model does not need to calculate gradients.

I also learned how to calculate classification accuracy by comparing the predicted classes with the actual labels.

Updated Learning Progress
12/15 Lessons Completed — 3 Lessons Remaining
 Lesson 1 — Tensors
 Lesson 2 — Autograd
 Lesson 3 — Neural Network Basics
 Lesson 4 — Activation Functions
 Lesson 5 — Loss Functions
 Lesson 6 — Training a Neural Network
 Lesson 7 — Optimizers
 Lesson 8 — Dataset and DataLoader
 Lesson 9 — Building Neural Networks with nn.Module
 Lesson 10 — Classification with PyTorch
 Lesson 11 — CNNs and Image Data
 Lesson 12 — CNN Training and Evaluation
 Lesson 13 — Transfer Learning
 Lesson 14 — Model Saving, Loading and Deployment Basics
 Lesson 15 — PyTorch Image Classification Project
---

## Goal

The goal of this repository is to build a strong understanding of PyTorch and deep learning fundamentals through consistent practice and progressively more advanced projects.
