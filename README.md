# PyTorch Learning

This repository contains my notes, practice code, and experiments while learning PyTorch and deep learning step by step.

The focus is on understanding core PyTorch concepts, implementing them through practical examples, and gradually progressing toward real-world deep learning projects.

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
* `.reshape()`

### What I Learned

Tensors are the fundamental data structure used by PyTorch. They are similar to arrays but are optimized for numerical computation and can also be used for GPU-based computation.

---

## Lesson 2 — Autograd

**File:** `autograd.py`

### Objectives

* Understand automatic differentiation.
* Learn how `requires_grad` works.
* Understand `.backward()`.
* Access gradients using `.grad`.
* Understand the computational graph.

### Key Concepts

* `requires_grad=True`
* Computational graph
* `.backward()`
* `.grad`

### What I Learned

PyTorch automatically tracks operations on tensors that require gradients. Calling `.backward()` calculates the gradients required for updating model parameters during training.

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

Activation functions introduce non-linearity into neural networks, allowing models to learn more complex patterns from data.

---

## Lesson 5 — Loss Functions

**File:** `loss_functions.py`

### Objectives

* Understand the purpose of a loss function.
* Learn how predictions are compared with actual values.
* Understand how loss is used during training.
* Learn common loss functions.

### Key Concepts

* Loss
* Prediction
* Target
* `nn.MSELoss()`
* `nn.CrossEntropyLoss()`

### What I Learned

A loss function measures how different a model's predictions are from the expected targets. The model uses this information during training to improve its parameters.

---

## Lesson 6 — Training a Neural Network

**File:** `training.py`

### Objectives

* Understand the basic training process.
* Learn the training loop.
* Perform forward propagation.
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
* Understand the learning rate.
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

Optimizers use gradients calculated during backpropagation to update model parameters. The learning rate controls the size of these parameter updates.

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

### What I Learned

A `Dataset` stores and provides access to training data, while a `DataLoader` makes it easier to load data in batches during training.

Using batches allows models to process groups of samples instead of loading the entire dataset at once.

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

PyTorch automatically tracks the weights and biases of these layers as model parameters.

---

## Lesson 10 — Classification with PyTorch

**File:** `classification.py`

### Objectives

* Understand classification in machine learning.
* Learn the difference between binary classification and regression.
* Build a neural network for binary classification.
* Understand class labels.
* Learn how `BCEWithLogitsLoss()` is used.
* Convert model outputs into probabilities.
* Convert probabilities into predicted classes.

### Key Concepts

* Binary classification
* Class labels
* Logits
* Probability
* Threshold
* `torch.sigmoid()`
* `nn.BCEWithLogitsLoss()`

### Important Concept — Logits

The final layer of a binary classification model produces a raw value called a **logit**.

A logit is not directly a probability.

It can be converted into a probability using:

```python
probability = torch.sigmoid(logit)
```

For example:

```text
Logit → Probability

 2.0 → 0.88
 0.0 → 0.50
-2.0 → 0.12
```

A common threshold is `0.5`:

```python
predicted_class = (probability >= 0.5).float()
```

### Loss Function

For binary classification:

```python
loss_function = nn.BCEWithLogitsLoss()
```

`BCEWithLogitsLoss()` combines sigmoid and binary cross-entropy in a numerically stable way.

Therefore, the model should return raw logits during training.

### What I Learned

Classification is used when the output represents a category rather than a continuous value.

For binary classification, the model produces one logit. After training, sigmoid can be used to convert the logit into a probability, followed by a threshold to obtain the predicted class.

---

## Lesson 11 — CNNs and Image Data

**File:** `cnn.py`

### Objectives

* Understand Convolutional Neural Networks.
* Learn why CNNs are useful for image data.
* Understand convolutional layers.
* Learn about pooling layers.
* Understand feature maps.
* Build a simple CNN using PyTorch.
* Understand image tensor shapes.

### Key Concepts

* CNN
* Convolution
* Kernel
* Feature map
* Channels
* `nn.Conv2d`
* `nn.MaxPool2d`
* `nn.Flatten()`

### Image Representation

A color image usually contains three channels:

```text
Red
Green
Blue
```

A single image can have the shape:

```text
[3, 64, 64]
```

For a batch of images:

```text
[Batch Size, Channels, Height, Width]
```

Example:

```text
[32, 3, 64, 64]
```

### CNN Architecture

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

### What I Learned

CNNs are designed to process data with spatial structure, especially images.

Convolutional layers learn visual features, while pooling layers reduce spatial dimensions and retain important information.

---

## Lesson 12 — CNN Training and Evaluation

**File:** `cnn_training.py`

### Objectives

* Learn how to train a CNN using image data.
* Understand the complete CNN training loop.
* Use `CrossEntropyLoss()` for multi-class classification.
* Train a model using an optimizer.
* Understand training loss and accuracy.
* Understand training and evaluation modes.
* Use `model.train()` and `model.eval()`.
* Evaluate a model using `torch.no_grad()`.
* Calculate classification accuracy.

### Key Concepts

* CNN training
* Training loop
* `CrossEntropyLoss()`
* `model.train()`
* `model.eval()`
* `torch.no_grad()`
* Accuracy
* Batch training
* Forward pass
* Backward pass
* Optimizer
* Evaluation

### Training Flow

```text
Training Data
     ↓
DataLoader
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
Updated Model
```

### Evaluation Flow

```text
Test Data
    ↓
CNN
    ↓
Predictions
    ↓
Compare with Labels
    ↓
Accuracy
```

During evaluation, model parameters are not updated.

### Training Mode

```python
model.train()
```

This puts the model into training mode.

This is important for layers such as:

* Dropout
* Batch Normalization

### Evaluation Mode

```python
model.eval()
```

This switches the model to evaluation mode.

During evaluation:

```python
with torch.no_grad():
    outputs = model(images)
```

Gradients are not calculated because the model is not being trained.

### Accuracy

Accuracy can be calculated using:

```python
predictions = outputs.argmax(dim=1)

correct = (predictions == labels).sum().item()

accuracy = correct / total
```

### What I Learned

A CNN requires separate training and evaluation phases.

During training, the model calculates predictions, loss, gradients, and updates its parameters.

During evaluation, the model makes predictions without updating its parameters.

I also learned how to calculate classification accuracy by comparing predicted classes with actual labels.

---

# Lesson 13 — Transfer Learning

**File:** `transfer_learning.py`

### Objectives

* Understand the concept of transfer learning.
* Learn why pretrained models are useful.
* Load a pretrained CNN.
* Understand pretrained weights.
* Freeze model parameters.
* Replace the final classification layer.
* Train a pretrained model on a new dataset.
* Understand fine-tuning.
* Learn the difference between feature extraction and fine-tuning.

---

## What is Transfer Learning?

Transfer learning is a technique where a model trained on one large dataset is reused for a different but related task.

Instead of training a CNN completely from scratch, we can start with a model that has already learned useful visual features.

### Without Transfer Learning

```text
Randomly Initialized CNN
        ↓
Train from Scratch
        ↓
Requires More Training
        ↓
Final Model
```

### With Transfer Learning

```text
Pretrained CNN
      ↓
Reuse Learned Features
      ↓
Replace Final Layer
      ↓
Train on New Dataset
      ↓
New Classification Model
```

Pretrained CNNs can already contain useful low-level and mid-level visual features such as:

```text
Edges
 ↓
Textures
 ↓
Shapes
 ↓
Patterns
```

These features can often be reused for another image classification task.

---

## Why Use Transfer Learning?

Training a deep CNN from scratch can require:

* Large datasets
* Significant computational resources
* More training time
* Careful model initialization

Transfer learning can reduce the amount of training required when working with a smaller dataset.

It is commonly used in practical computer vision applications.

---

## Pretrained Models in PyTorch

PyTorch provides several pretrained computer vision models.

Examples include:

* ResNet
* VGG
* DenseNet
* EfficientNet
* MobileNet

For this lesson, I use **ResNet18** as an example.

```python
from torchvision import models

model = models.resnet18(weights="DEFAULT")
```

The pretrained model already contains learned parameters.

---

## Freezing Model Parameters

If the goal is to use the pretrained model as a feature extractor, the existing parameters can be frozen:

```python
for parameter in model.parameters():
    parameter.requires_grad = False
```

This prevents the pretrained layers from being updated during training.

The model can then focus on learning the new classification layer.

---

## Replacing the Final Layer

A pretrained ResNet18 was originally designed for a specific number of classes.

For a new classification problem, the final layer can be replaced.

```python
import torch.nn as nn

model.fc = nn.Linear(
    model.fc.in_features,
    2
)
```

Here:

```text
model.fc.in_features
```

gets the number of input features expected by the original final layer.

The new layer produces:

```text
2 outputs
```

for a two-class classification problem.

---

## Transfer Learning Architecture

```text
Input Image
     ↓
Pretrained CNN
     ↓
Learned Visual Features
     ↓
Frozen Layers
     ↓
New Classification Layer
     ↓
Class Prediction
```

For example:

```text
Image
  ↓
ResNet18
  ↓
Feature Extraction
  ↓
Fully Connected Layer
  ↓
Class 0 / Class 1
```

---

## Feature Extraction vs Fine-Tuning

There are two common approaches to transfer learning.

### Feature Extraction

The pretrained layers are frozen:

```python
for parameter in model.parameters():
    parameter.requires_grad = False
```

Only the new classification layer is trained.

```text
Pretrained Layers → Frozen
Classification Layer → Trainable
```

This is useful when the new dataset is relatively small or similar to the original training domain.

### Fine-Tuning

Instead of freezing every pretrained layer, some or all layers can be allowed to update.

```text
Pretrained Model
      ↓
Selected Layers
      ↓
Updated During Training
      ↓
New Task
```

Fine-tuning allows the model to adapt its learned features to the new dataset.

---

## Example

```python
import torch
from torch import nn
from torchvision import models


# --------------------------------------------------
# 1. Load pretrained model
# --------------------------------------------------

model = models.resnet18(weights="DEFAULT")


# --------------------------------------------------
# 2. Freeze pretrained parameters
# --------------------------------------------------

for parameter in model.parameters():
    parameter.requires_grad = False


# --------------------------------------------------
# 3. Replace final classification layer
# --------------------------------------------------

model.fc = nn.Linear(
    model.fc.in_features,
    2
)


# --------------------------------------------------
# 4. Create loss function
# --------------------------------------------------

loss_function = nn.CrossEntropyLoss()


# --------------------------------------------------
# 5. Create optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)


# --------------------------------------------------
# 6. Example input
# --------------------------------------------------

images = torch.randn(4, 3, 224, 224)

outputs = model(images)

print(model)
print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
```

### Expected Output Shape

The input has:

```text
[4, 3, 224, 224]
```

which means:

```text
4   → images in the batch
3   → RGB channels
224 → image height
224 → image width
```

The output has:

```text
[4, 2]
```

which means:

```text
4 → predictions for 4 images
2 → two class outputs
```

---

## Transfer Learning Training Flow

```text
Image Dataset
      ↓
DataLoader
      ↓
Pretrained ResNet18
      ↓
Feature Extraction
      ↓
New Classification Layer
      ↓
Logits
      ↓
CrossEntropyLoss
      ↓
Backward Pass
      ↓
Optimizer
      ↓
Updated Classification Layer
```

---

## Fine-Tuning Flow

```text
Pretrained Model
      ↓
Load Learned Weights
      ↓
Replace Classification Layer
      ↓
Unfreeze Selected Layers
      ↓
Train with Small Learning Rate
      ↓
Adapt Model to New Dataset
```

---

## Important Consideration — Input Images

Pretrained image models usually expect images in a specific format.

For ResNet-style models, images are commonly resized and normalized before being passed to the network.

A typical preprocessing pipeline can include:

```text
Original Image
      ↓
Resize
      ↓
Convert to Tensor
      ↓
Normalize
      ↓
CNN
```

Example:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
```

For pretrained models, normalization using the expected pretrained-model statistics is also important.

---

## Key Concepts

* Transfer learning
* Pretrained model
* Pretrained weights
* Feature extraction
* Fine-tuning
* Frozen parameters
* `requires_grad`
* ResNet
* `torchvision.models`
* `nn.Linear`
* Image preprocessing

---

## What I Learned

Transfer learning allows a pretrained neural network to be adapted to a new image classification task.

Instead of learning every visual feature from the beginning, the model can reuse features learned from a large dataset.

I learned how to load a pretrained ResNet model, freeze its parameters, replace its final classification layer, and prepare it for a new classification task.

I also learned the difference between **feature extraction** and **fine-tuning**.

Transfer learning is an important technique for practical computer vision applications because it can reduce training requirements and make it easier to build models when the available dataset is limited.

---

# Updated Learning Progress

**13/15 Lessons Completed — 2 Lessons Remaining**

* [x] Lesson 1 — Tensors
* [x] Lesson 2 — Autograd
* [x] Lesson 3 — Neural Network Basics
* [x] Lesson 4 — Activation Functions
* [x] Lesson 5 — Loss Functions
* [x] Lesson 6 — Training a Neural Network
* [x] Lesson 7 — Optimizers
* [x] Lesson 8 — Dataset and DataLoader
* [x] Lesson 9 — Building Neural Networks with `nn.Module`
* [x] Lesson 10 — Classification with PyTorch
* [x] Lesson 11 — CNNs and Image Data
* [x] Lesson 12 — CNN Training and Evaluation
* [x] Lesson 13 — Transfer Learning
* [ ] Lesson 14 — Model Saving, Loading and Deployment Basics
* [ ] Lesson 15 — PyTorch Image Classification Project

---

# Goal

The goal of this repository is to build a strong understanding of PyTorch and deep learning fundamentals through consistent practice and progressively more advanced projects.

The learning path is designed to progress from basic tensor operations to neural networks, CNNs, transfer learning, model deployment, and finally a complete image classification project.

By the end of the learning path, the goal is to have practical experience with:

* PyTorch fundamentals
* Neural networks
* Training and evaluation
* CNNs
* Image classification
* Transfer learning
* Model saving and loading
* Basic deployment
* End-to-end deep learning projects
