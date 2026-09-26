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

## Learning Progress

**9/15 Lessons Completed — 6 Lessons Remaining**

* [x] Lesson 1 — Tensors
* [x] Lesson 2 — Autograd
* [x] Lesson 3 — Neural Network Basics
* [x] Lesson 4 — Activation Functions
* [x] Lesson 5 — Loss Functions
* [x] Lesson 6 — Training a Neural Network
* [x] Lesson 7 — Optimizers
* [x] Lesson 8 — Dataset and DataLoader
* [x] Lesson 9 — Building Neural Networks with `nn.Module`
* [ ] Lesson 10 — Classification with PyTorch
* [ ] Lesson 11 — CNNs and Image Data
* [ ] Lesson 12 — CNN Training and Evaluation
* [ ] Lesson 13 — Transfer Learning
* [ ] Lesson 14 — Model Saving, Loading and Deployment Basics
* [ ] Lesson 15 — PyTorch Image Classification Project

---

## Tools and Technologies

* Python
* PyTorch
* NumPy
* VS Code
* Git
* GitHub

---

## Goal

The goal of this repository is to build a strong understanding of PyTorch and deep learning fundamentals through consistent practice and progressively more advanced projects.
