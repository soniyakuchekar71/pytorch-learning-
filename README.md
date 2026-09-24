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

## Lesson 7 — Optimizers

**File:** `optimizers.py`

### Objectives

* Understand what an optimizer is.
* Understand why optimizers are used when training neural networks.
* Learn how optimizers update model parameters.
* Understand the difference between manually updating parameters and using an optimizer.
* Learn how to use `torch.optim`.
* Understand `SGD` and the basic idea of `Adam`.

### Topics Covered

* Optimizers
* `torch.optim`
* `SGD`
* `Adam`
* Learning rate
* `optimizer.zero_grad()`
* `optimizer.step()`
* Parameter updates
* Optimizer and loss functions

### What is an Optimizer?

An optimizer is responsible for updating the parameters of a neural network using the gradients calculated during backpropagation.

Previously, in Lesson 3, the parameter update was done manually:

```text
new value = old value - learning rate × gradient
```

An optimizer performs this parameter update automatically.

### Training Process

A basic training process looks like:

```text
Input
  ↓
Neural Network
  ↓
Prediction
  ↓
Loss
  ↓
backward()
  ↓
Gradients
  ↓
Optimizer
  ↓
Parameter Update
```

### Why Do We Need Optimizers?

Without an optimizer, we would need to manually update every parameter:

```python
with torch.no_grad():
    parameter -= learning_rate * parameter.grad
```

For a real neural network containing thousands or millions of parameters, doing this manually would be difficult.

Optimizers handle these updates for us.

---

### SGD

SGD stands for **Stochastic Gradient Descent**.

It is one of the simplest optimization algorithms.

In PyTorch:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

Here:

* `model.parameters()` gives the parameters that need to be updated.
* `lr` means learning rate.
* `0.01` is the learning rate.

### Adam

Adam is another commonly used optimizer.

In PyTorch:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Adam automatically adjusts how parameters are updated based on gradient information.

---

## `optimizer.zero_grad()`

Gradients can accumulate in PyTorch.

Before calculating gradients for the next training step, we normally clear the old gradients:

```python
optimizer.zero_grad()
```

### Why?

The training cycle is:

```text
Clear old gradients
       ↓
Forward pass
       ↓
Calculate loss
       ↓
Backward pass
       ↓
Update parameters
```

---

## `optimizer.step()`

After calculating gradients using:

```python
loss.backward()
```

we use:

```python
optimizer.step()
```

to update the model parameters.

So:

```python
loss.backward()
```

calculates the gradients.

```python
optimizer.step()
```

uses those gradients to update the parameters.

---

## Example

```python
import torch
import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        return self.layer(x)


model = SimpleModel()

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


x = torch.tensor([[2.0]])
target = torch.tensor([[4.0]])


for step in range(10):

    optimizer.zero_grad()

    prediction = model(x)

    loss = loss_function(prediction, target)

    loss.backward()

    optimizer.step()

    print(
        "Step:",
        step + 1,
        "Prediction:",
        prediction.item(),
        "Loss:",
        loss.item()
    )
```

### What Happens in This Example?

#### Step 1 — Create the model

```python
model = SimpleModel()
```

The model contains a linear layer.

---

#### Step 2 — Create the loss function

```python
loss_function = nn.MSELoss()
```

This measures the difference between the prediction and target.

---

#### Step 3 — Create the optimizer

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)
```

The optimizer will update the model's parameters.

---

#### Step 4 — Clear previous gradients

```python
optimizer.zero_grad()
```

This removes gradients from the previous training step.

---

#### Step 5 — Forward pass

```python
prediction = model(x)
```

The input goes through the neural network.

---

#### Step 6 — Calculate loss

```python
loss = loss_function(prediction, target)
```

This tells us how far the prediction is from the target.

---

#### Step 7 — Calculate gradients

```python
loss.backward()
```

PyTorch calculates how each model parameter contributed to the loss.

---

#### Step 8 — Update parameters

```python
optimizer.step()
```

The optimizer uses the gradients to update the model parameters.

---

## Complete Training Cycle

The most important pattern to remember is:

```python
optimizer.zero_grad()

prediction = model(x)

loss = loss_function(prediction, target)

loss.backward()

optimizer.step()
```

This pattern will appear repeatedly when training PyTorch models.

### Important Difference

Previously:

```python
x -= learning_rate * x.grad
```

We manually updated the parameter.

Now:

```python
optimizer.step()
```

The optimizer performs the update.

---

## SGD vs Adam

| Optimizer | Description                                       |
| --------- | ------------------------------------------------- |
| `SGD`     | Simple gradient-based optimizer                   |
| `Adam`    | Adaptive optimizer that adjusts parameter updates |

Example:

```python
torch.optim.SGD(model.parameters(), lr=0.01)
```

```python
torch.optim.Adam(model.parameters(), lr=0.001)
```

For now, the important thing is understanding **how an optimizer fits into the training process** rather than memorizing every optimizer.

---

## Learning Rate

The learning rate controls how large the parameter updates are.

```python
lr=0.01
```

A very large learning rate can cause training to move too aggressively.

A very small learning rate can make training very slow.

```text
Learning Rate
      ↓
Size of Parameter Update
      ↓
Training Behavior
```

---

## Key Concepts to Remember

### `optimizer.zero_grad()`

Clears previous gradients.

### `loss.backward()`

Calculates gradients.

### `optimizer.step()`

Updates model parameters.

### `lr`

Controls the learning rate.

### `model.parameters()`

Provides the parameters that the optimizer should update.

---

## Learning Outcomes

After completing this lesson, I can:

* Explain what an optimizer does.
* Understand why optimizers are needed for neural network training.
* Create an optimizer using `torch.optim`.
* Use SGD in PyTorch.
* Understand the basic idea of Adam.
* Use `optimizer.zero_grad()`.
* Use `optimizer.step()`.
* Understand how loss, gradients, and optimizers work together.
* Explain the basic PyTorch training cycle.

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
| 7      | Optimizers                      | Completed |

**Progress: 7/7 lessons completed**

---

# Concepts Learned So Far

The first seven lessons build the foundation for understanding how PyTorch models learn.

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
   ↓
Optimizer
   ↓
Parameter Update
```

So far, I have learned how PyTorch represents data using tensors, calculates gradients using Autograd, updates parameters through gradient descent, creates neural networks using `nn.Module`, introduces non-linearity using activation functions, measures prediction error using loss functions, and uses optimizers to automatically update model parameters.

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
├── loss_functions.py
└── optimizers.py
```

---

# Next Topics

The upcoming lessons will focus on:

* Training loops
* Training a complete neural network
* Dataset and DataLoader
* Batches
* Epochs
* Model evaluation
* Training and validation
* Saving and loading models
* Building practical deep learning models

---

# Goal

Build a strong foundation in PyTorch and gradually progress from basic tensor operations to implementing and training practical deep learning models.
