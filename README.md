# PyTorch Learning

This repository contains my notes, practice code, and experiments while learning PyTorch from the basics.

The focus is on understanding the core concepts step by step and applying them through small practical examples.

---

## Lesson 1 — Tensors

**File:** `tensors.py`

### Objectives

* Understand what tensors are and why they are used in PyTorch.
* Learn how to create tensors.
* Understand tensor dimensions, shapes, and data types.
* Perform basic operations on tensors.

### Topics Covered

* Creating tensors
* Tensor dimensions and shapes
* Data types
* Basic tensor operations
* Arithmetic operations
* Accessing tensor values

### Learning Outcomes

After completing this lesson, I can:

* Create tensors using PyTorch.
* Check the shape and data type of a tensor.
* Perform basic mathematical operations on tensors.
* Understand the role of tensors in PyTorch.

---

## Lesson 2 — Autograd

**File:** `autograd.py`

### Objectives

* Understand automatic differentiation in PyTorch.
* Learn how PyTorch tracks operations for gradient calculation.
* Understand how to calculate gradients using Autograd.
* Understand the purpose of `requires_grad`, `backward()`, and `grad`.

### Topics Covered

* Automatic differentiation
* `requires_grad=True`
* `backward()`
* `grad`
* Basic gradient calculation
* Gradients with multiple variables

### Example

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)
```

### Learning Outcomes

After completing this lesson, I can:

* Enable gradient tracking for tensors.
* Calculate gradients using `.backward()`.
* Access calculated gradients using `.grad`.
* Understand why gradients are important when training neural networks.

---

## Lesson 3 — Gradient Descent

**File:** `gradient_descent.py`

### Objectives

* Understand the basic idea behind gradient descent.
* Learn how gradients are used to update parameters.
* Understand the role of the learning rate.
* Apply gradient descent using PyTorch.

### Topics Covered

* Gradient descent
* Learning rate
* Parameter updates
* `torch.no_grad()`
* `zero_()`
* Using Autograd with gradient descent

### Basic Update Rule

```text
new value = old value - learning rate × gradient
```

### Example

```python
with torch.no_grad():
    x -= learning_rate * x.grad

x.grad.zero_()
```

### Learning Outcomes

After completing this lesson, I can:

* Explain the basic idea of gradient descent.
* Understand how a gradient affects a parameter update.
* Use a learning rate to control parameter updates.
* Update parameters while avoiding gradient tracking.
* Clear previously calculated gradients.

---

## Lesson 4 — Neural Networks and `nn.Module`

**File:** `neural_network.py`

### Objectives

* Understand the basic structure of a neural network.
* Learn how PyTorch represents neural network models.
* Understand the purpose of `nn.Module`.
* Learn how to create a simple neural network.
* Understand layers, inputs, outputs, and forward propagation.

### Topics Covered

* Neural network basics
* `torch.nn`
* `nn.Module`
* `nn.Linear`
* Model parameters
* Forward propagation
* Input and output dimensions
* `forward()` method

### Basic Neural Network Structure

```text
Input
  ↓
Linear Layer
  ↓
Output
```

A neural network takes input data, processes it through layers, and produces an output.

### Example

```python
import torch
import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)


model = SimpleModel()

x = torch.tensor([[2.0, 3.0]])

output = model(x)

print(output)
```

### Key Concepts

#### `nn.Module`

`nn.Module` is the base class used to create neural network models in PyTorch.

```python
class SimpleModel(nn.Module):
```

It allows PyTorch to keep track of the model's parameters and layers.

#### `nn.Linear`

```python
nn.Linear(2, 1)
```

This creates a fully connected layer with:

* 2 input features
* 1 output feature

#### `forward()`

The `forward()` method defines how the input moves through the model.

```python
def forward(self, x):
    return self.layer(x)
```

#### Calling the Model

```python
output = model(x)
```

PyTorch automatically uses the model's `forward()` method when the model is called.

### Learning Outcomes

After completing this lesson, I can:

* Explain the basic structure of a neural network.
* Create a neural network using `nn.Module`.
* Create layers using `nn.Linear`.
* Define a `forward()` method.
* Pass input data through a neural network.
* Understand the relationship between inputs, layers, and outputs.

---

## Overall Learning Progress

| Lesson | Topic                           | Status    |
| ------ | ------------------------------- | --------- |
| 1      | Tensors                         | Completed |
| 2      | Autograd                        | Completed |
| 3      | Gradient Descent                | Completed |
| 4      | Neural Networks and `nn.Module` | Completed |

**Progress: 4 lessons completed**

---

## Concepts Learned So Far

The first four lessons build the foundation for understanding how PyTorch models learn:

```text
Tensor
   ↓
Calculation
   ↓
Gradient
   ↓
Parameter Update
   ↓
Neural Network
   ↓
Model Output
```

So far, I have learned how PyTorch represents data using tensors, calculates gradients using Autograd, updates parameters through gradient descent, and creates basic neural network models using `nn.Module`.

---

## Repository Structure

```text
pytorch-learning/
│
├── README.md
├── tensors.py
├── autograd.py
├── gradient_descent.py
└── neural_network.py
```

---

## Next Topics

The upcoming lessons will focus on:

* Activation functions
* Loss functions
* Optimizers
* Training a neural network
* Model evaluation
* Training and validation
* Building practical neural networks

---

## Goal

Build a strong foundation in PyTorch and gradually progress from basic tensor operations to implementing and training practical deep learning models.
