# PyTorch Learning

This repository contains my notes, practice code, and experiments while learning PyTorch from the basics.

The focus is on understanding core concepts step by step and applying them through small practical examples.

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
* Reshaping tensors

### Example

```python
import torch

x = torch.tensor([10, 20, 30, 40])

print("Tensor:", x)
print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Data type:", x.dtype)

x = x.reshape(2, 2)

print("Reshaped tensor:")
print(x)
```

### Learning Outcomes

After completing this lesson, I can:

* Create tensors using PyTorch.
* Check the shape and data type of a tensor.
* Understand tensor dimensions.
* Perform basic mathematical operations.
* Reshape tensors.
* Access tensor values.

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
* Gradient calculation
* Computational graphs

### Example

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print("x:", x)
print("y:", y)
print("Gradient:", x.grad)
```

### Learning Outcomes

After completing this lesson, I can:

* Enable gradient tracking for tensors.
* Calculate gradients using `.backward()`.
* Access calculated gradients using `.grad`.
* Understand the basic idea of computational graphs.
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
* Autograd with gradient descent

### Basic Update Rule

```text
new value = old value - learning rate × gradient
```

### Example

```python
import torch

x = torch.tensor(5.0, requires_grad=True)

learning_rate = 0.1

for step in range(10):

    y = x ** 2

    y.backward()

    with torch.no_grad():
        x -= learning_rate * x.grad

    x.grad.zero_()

    print("Step:", step + 1, "x:", x.item())
```

### Learning Outcomes

After completing this lesson, I can:

* Explain the basic idea of gradient descent.
* Understand how gradients affect parameter updates.
* Understand the purpose of the learning rate.
* Update parameters using gradients.
* Use `torch.no_grad()` during parameter updates.
* Clear gradients using `.zero_()`.

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

print("Input:", x)
print("Output:", output)
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

## Lesson 5 — Activation Functions

**File:** `activation_functions.py`

### Objectives

* Understand why activation functions are used in neural networks.
* Learn how activation functions introduce non-linearity.
* Understand commonly used activation functions in PyTorch.
* Learn how to apply activation functions to tensors.
* Understand the role of activation functions between neural network layers.

### Topics Covered

* Activation functions
* Non-linearity
* ReLU
* Sigmoid
* Tanh
* `nn.ReLU`
* `nn.Sigmoid`
* `nn.Tanh`

### Why Activation Functions?

Activation functions allow neural networks to learn complex patterns.

Without activation functions, multiple linear layers would still behave like a linear transformation.

Activation functions introduce non-linearity into the network.

### ReLU

ReLU stands for Rectified Linear Unit.

```text
ReLU(x) = max(0, x)
```

It converts negative values to `0` and keeps positive values unchanged.

### Sigmoid

Sigmoid converts values into a range between `0` and `1`.

It is commonly used for binary classification outputs.

### Tanh

Tanh converts values into a range between `-1` and `1`.

### Example

```python
import torch
import torch.nn as nn

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()

print("Input:", x)
print("ReLU:", relu(x))
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
```

### Activation Function Flow

```text
Input
  ↓
Linear Layer
  ↓
Activation Function
  ↓
Next Layer
  ↓
Output
```

### Learning Outcomes

After completing this lesson, I can:

* Explain why activation functions are needed.
* Understand the concept of non-linearity.
* Use ReLU, Sigmoid, and Tanh in PyTorch.
* Apply activation functions to tensors.
* Understand where activation functions are used in neural networks.

---

## Lesson 6 — Loss Functions

**File:** `loss_functions.py`

### Objectives

* Understand what a loss function is.
* Understand why neural networks need loss functions.
* Learn how PyTorch calculates prediction error.
* Learn how to use `nn.MSELoss()`.
* Understand the relationship between predictions, targets, and loss.
* Understand why minimizing loss is important during training.

### Topics Covered

* Loss functions
* Prediction vs actual value
* Model error
* Mean Squared Error
* `nn.MSELoss()`
* Loss calculation
* Loss minimization

### What is a Loss Function?

A loss function measures the difference between the model's prediction and the actual target.

```text
Input
  ↓
Neural Network
  ↓
Prediction
  ↓
Loss Function
  ↓
Loss
```

The training process tries to reduce this loss.

### Mean Squared Error

Mean Squared Error calculates the average squared difference between predictions and actual values.

```text
MSE = average((prediction - actual)²)
```

### Example

```python
import torch
import torch.nn as nn


prediction = torch.tensor([2.5])
actual = torch.tensor([3.0])


mse_loss = nn.MSELoss()

loss = mse_loss(prediction, actual)


print("Prediction:", prediction)
print("Actual:", actual)
print("MSE Loss:", loss)
```

### Example Calculation

```text
Prediction = 2.5
Actual     = 3.0

Difference = 2.5 - 3.0
           = -0.5

Squared difference = 0.25
```

Therefore, the loss for this single value is:

```text
0.25
```

### Common PyTorch Loss Functions

| Loss Function           | Common Use                 |
| ----------------------- | -------------------------- |
| `nn.MSELoss()`          | Regression                 |
| `nn.L1Loss()`           | Regression                 |
| `nn.CrossEntropyLoss()` | Multi-class classification |
| `nn.BCELoss()`          | Binary classification      |

### Loss and Training

```text
Prediction
     ↓
Loss Function
     ↓
Loss
     ↓
Gradient
     ↓
Parameter Update
     ↓
Better Prediction
```

### Learning Outcomes

After completing this lesson, I can:

* Explain what a loss function does.
* Explain the difference between prediction and target.
* Calculate prediction error using MSE.
* Use `nn.MSELoss()` in PyTorch.
* Explain why training tries to minimize loss.
* Understand how loss connects predictions to gradient-based learning.

---

# Overall Learning Progress

| Lesson | Topic                           | Status    |
| ------ | ------------------------------- | --------- |
| 1      | Tensors                         | Completed |
| 2      | Autograd                        | Completed |
| 3      | Gradient Descent                | Completed |
| 4      | Neural Networks and `nn.Module` | Completed |
| 5      | Activation Functions            | Completed |
| 6      | Loss Functions                  | Completed |

**Progress: 6/6 lessons completed**

---

# Concepts Learned So Far

The first six lessons build the foundation for understanding how PyTorch models learn.

```text
Tensor
   ↓
Autograd
   ↓
Gradient
   ↓
Parameter Update
   ↓
Neural Network
   ↓
Activation Function
   ↓
Prediction
   ↓
Loss
```

So far, I have learned how PyTorch represents data using tensors, calculates gradients using Autograd, updates parameters through gradient descent, creates neural networks using `nn.Module`, introduces non-linearity using activation functions, and measures prediction error using loss functions.

---

# Repository Structure

```text
pytorch-learning/
│
├── README.md
├── tensors.py
├── autograd.py
├── gradient_descent.py
├── neural_network.py
├── activation_functions.py
└── loss_functions.py
```

---

# Next Topics

The upcoming lessons will focus on:

* Optimizers
* SGD
* Adam
* Training a neural network
* Training loops
* Model evaluation
* Training and validation
* Saving and loading models
* Building practical deep learning models

---

# Goal

Build a strong foundation in PyTorch and gradually progress from basic tensor operations to implementing and training practical deep learning models.
