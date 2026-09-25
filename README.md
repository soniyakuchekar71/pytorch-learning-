PyTorch Learning

This repository contains my notes, practice code, and experiments while learning PyTorch from the basics.

The focus is on understanding core concepts step by step and applying them through small practical examples.

Lesson 1 — Tensors

File: tensors.py

Objectives

Understand what tensors are and why they are used in PyTorch.

Learn how to create tensors.

Understand tensor dimensions, shapes, and data types.

Perform basic operations on tensors.

Topics Covered

Creating tensors

Tensor dimensions and shapes

Data types

Basic tensor operations

Arithmetic operations

Accessing tensor values

Reshaping tensors

Example

import torch

x = torch.tensor([10, 20, 30, 40])

print("Tensor:", x)
print("Shape:", x.shape)
print("Dimensions:", x.ndim)
print("Data type:", x.dtype)

x = x.reshape(2, 2)

print("Reshaped tensor:")
print(x)

Learning Outcomes

After completing this lesson, I can:

Create tensors using PyTorch.

Check the shape and data type of a tensor.

Understand tensor dimensions.

Perform basic mathematical operations.

Reshape tensors.

Access tensor values.

Lesson 2 — Autograd

File: autograd.py

Objectives

Understand automatic differentiation in PyTorch.

Learn how PyTorch tracks operations for gradient calculation.

Understand how to calculate gradients using Autograd.

Understand the purpose of requires_grad, backward(), and grad.

Topics Covered

Automatic differentiation

requires_grad=True

backward()

grad

Gradient calculation

Computational graphs

Example

import torch

x = torch.tensor(3.0, requires_grad=True)

y = x ** 2

y.backward()

print("x:", x)
print("y:", y)
print("Gradient:", x.grad)

Learning Outcomes

After completing this lesson, I can:

Enable gradient tracking for tensors.

Calculate gradients using .backward().

Access calculated gradients using .grad.

Understand the basic idea of computational graphs.

Understand why gradients are important when training neural networks.

Lesson 3 — Gradient Descent

File: gradient_descent.py

Objectives

Understand the basic idea behind gradient descent.

Learn how gradients are used to update parameters.

Understand the role of the learning rate.

Apply gradient descent using PyTorch.

Topics Covered

Gradient descent

Learning rate

Parameter updates

torch.no_grad()

zero_()

Autograd with gradient descent

Basic Update Rule

new value = old value - learning rate × gradient

Example

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

Learning Outcomes

After completing this lesson, I can:

Explain the basic idea of gradient descent.

Understand how gradients affect parameter updates.

Understand the purpose of the learning rate.

Update parameters using gradients.

Use torch.no_grad() during parameter updates.

Clear gradients using .zero_().

Lesson 4 — Neural Networks and nn.Module

File: neural_network.py

Objectives

Understand the basic structure of a neural network.

Learn how PyTorch represents neural network models.

Understand the purpose of nn.Module.

Learn how to create a simple neural network.

Understand layers, inputs, outputs, and forward propagation.

Topics Covered

Neural network basics

torch.nn

nn.Module

nn.Linear

Model parameters

Forward propagation

Input and output dimensions

forward() method

Basic Neural Network Structure

Input
  ↓
Linear Layer
  ↓
Output

Example

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

Key Concepts

nn.Module

nn.Module is the base class used to create neural network models in PyTorch.

class SimpleModel(nn.Module):

It allows PyTorch to keep track of the model's parameters and layers.

nn.Linear

nn.Linear(2, 1)

This creates a fully connected layer with:

2 input features

1 output feature

forward()

The forward() method defines how the input moves through the model.

def forward(self, x):
    return self.layer(x)

Calling the Model

output = model(x)

PyTorch automatically uses the model's forward() method when the model is called.

Learning Outcomes

After completing this lesson, I can:

Explain the basic structure of a neural network.

Create a neural network using nn.Module.

Create layers using nn.Linear.

Define a forward() method.

Pass input data through a neural network.

Understand the relationship between inputs, layers, and outputs.

Lesson 5 — Activation Functions

File: activation_functions.py

Objectives

Understand why activation functions are used in neural networks.

Learn how activation functions introduce non-linearity.

Understand commonly used activation functions in PyTorch.

Learn how to apply activation functions to tensors.

Understand the role of activation functions between neural network layers.

Topics Covered

Activation functions

Non-linearity

ReLU

Sigmoid

Tanh

nn.ReLU

nn.Sigmoid

nn.Tanh

Why Activation Functions?

Activation functions allow neural networks to learn complex patterns.

Without activation functions, multiple linear layers would still behave like a linear transformation.

Activation functions introduce non-linearity into the network.

ReLU

ReLU stands for Rectified Linear Unit.

ReLU(x) = max(0, x)

It converts negative values to 0 and keeps positive values unchanged.

Sigmoid

Sigmoid converts values into a range between 0 and 1.

It is commonly used for binary classification outputs.

Tanh

Tanh converts values into a range between -1 and 1.

Example

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

Activation Function Flow

Input
  ↓
Linear Layer
  ↓
Activation Function
  ↓
Next Layer
  ↓
Output

Learning Outcomes

After completing this lesson, I can:

Explain why activation functions are needed.

Understand the concept of non-linearity.

Use ReLU, Sigmoid, and Tanh in PyTorch.

Apply activation functions to tensors.

Understand where activation functions are used in neural networks.

Lesson 6 — Loss Functions

File: loss_functions.py

Objectives

Understand what a loss function is.

Understand why neural networks need loss functions.

Learn how PyTorch calculates prediction error.

Learn how to use nn.MSELoss().

Understand the relationship between predictions, targets, and loss.

Understand why minimizing loss is important during training.

Topics Covered

Loss functions

Prediction vs actual value

Model error

Mean Squared Error

nn.MSELoss()

Loss calculation

Loss minimization

What is a Loss Function?

A loss function measures the difference between the model's prediction and the actual target.

Input
  ↓
Neural Network
  ↓
Prediction
  ↓
Loss Function
  ↓
Loss

The training process tries to reduce this loss.

Mean Squared Error

Mean Squared Error calculates the average squared difference between predictions and actual values.

MSE = average((prediction - actual)²)

Example

import torch
import torch.nn as nn


prediction = torch.tensor([2.5])
actual = torch.tensor([3.0])


mse_loss = nn.MSELoss()

loss = mse_loss(prediction, actual)


print("Prediction:", prediction)
print("Actual:", actual)
print("MSE Loss:", loss)

Example Calculation

Prediction = 2.5
Actual     = 3.0

Difference = 2.5 - 3.0
           = -0.5

Squared difference = 0.25

Therefore, the loss for this single value is:

0.25

Common PyTorch Loss Functions

Loss Function

Common Use

nn.MSELoss()

Regression

nn.L1Loss()

Regression

nn.CrossEntropyLoss()

Multi-class classification

nn.BCELoss()

Binary classification

Loss and Training

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

Learning Outcomes

After completing this lesson, I can:

Explain what a loss function does.

Explain the difference between prediction and target.

Calculate prediction error using MSE.

Use nn.MSELoss() in PyTorch.

Explain why training tries to minimize loss.

Understand how loss connects predictions to gradient-based learning.

Lesson 7 — Optimizers

File: optimizers.py

Objectives

Understand what an optimizer is.

Understand why optimizers are used when training neural networks.

Learn how optimizers update model parameters.

Understand the difference between manually updating parameters and using an optimizer.

Learn how to use torch.optim.

Understand SGD and the basic idea of Adam.

Topics Covered

Optimizers

torch.optim

SGD

Adam

Learning rate

optimizer.zero_grad()

optimizer.step()

Parameter updates

Optimizer and loss functions

What is an Optimizer?

An optimizer is responsible for updating the parameters of a neural network using the gradients calculated during backpropagation.

Previously, in Lesson 3, the parameter update was done manually:

new value = old value - learning rate × gradient

An optimizer performs this parameter update automatically.

Training Process

A basic training process looks like:

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

Why Do We Need Optimizers?

Without an optimizer, we would need to manually update every parameter:

with torch.no_grad():
    parameter -= learning_rate * parameter.grad

For a real neural network containing thousands or millions of parameters, doing this manually would be difficult.

Optimizers handle these updates for us.

SGD

SGD stands for Stochastic Gradient Descent.

It is one of the simplest optimization algorithms.

In PyTorch:

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

Here:

model.parameters() gives the parameters that need to be updated.

lr means learning rate.

0.01 is the learning rate.

Adam

Adam is another commonly used optimizer.

In PyTorch:

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

Adam automatically adjusts how parameters are updated based on gradient information.

optimizer.zero_grad()

Gradients can accumulate in PyTorch.

Before calculating gradients for the next training step, we normally clear the old gradients:

optimizer.zero_grad()

Why?

The training cycle is:

Clear old gradients
       ↓
Forward pass
       ↓
Calculate loss
       ↓
Backward pass
       ↓
Update parameters

optimizer.step()

After calculating gradients using:

loss.backward()

we use:

optimizer.step()

to update the model parameters.

So:

loss.backward()

calculates the gradients.

optimizer.step()

uses those gradients to update the parameters.

Example

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

What Happens in This Example?

Step 1 — Create the model

model = SimpleModel()

The model contains a linear layer.

Step 2 — Create the loss function

loss_function = nn.MSELoss()

This measures the difference between the prediction and target.

Step 3 — Create the optimizer

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

The optimizer will update the model's parameters.

Step 4 — Clear previous gradients

optimizer.zero_grad()

This removes gradients from the previous training step.

Step 5 — Forward pass

prediction = model(x)

The input goes through the neural network.

Step 6 — Calculate loss

loss = loss_function(prediction, target)

This tells us how far the prediction is from the target.

Step 7 — Calculate gradients

loss.backward()

PyTorch calculates how each model parameter contributed to the loss.

Step 8 — Update parameters

optimizer.step()

The optimizer uses the gradients to update the model parameters.

Complete Training Cycle

The most important pattern to remember is:

optimizer.zero_grad()

prediction = model(x)

loss = loss_function(prediction, target)

loss.backward()

optimizer.step()

This pattern will appear repeatedly when training PyTorch models.

Important Difference

Previously:

x -= learning_rate * x.grad

We manually updated the parameter.

Now:

optimizer.step()

The optimizer performs the update.

SGD vs Adam

Optimizer

Description

SGD

Simple gradient-based optimizer

Adam

Adaptive optimizer that adjusts parameter updates

Example:

torch.optim.SGD(model.parameters(), lr=0.01)

torch.optim.Adam(model.parameters(), lr=0.001)

For now, the important thing is understanding how an optimizer fits into the training process rather than memorizing every optimizer.

Learning Rate

The learning rate controls how large the parameter updates are.

lr=0.01

A very large learning rate can cause training to move too aggressively.

A very small learning rate can make training very slow.

Learning Rate
      ↓
Size of Parameter Update
      ↓
Training Behavior

Key Concepts to Remember

optimizer.zero_grad()

Clears previous gradients.

loss.backward()

Calculates gradients.

optimizer.step()

Updates model parameters.

lr

Controls the learning rate.

model.parameters()

Provides the parameters that the optimizer should update.

Learning Outcomes

After completing this lesson, I can:

Explain what an optimizer does.

Understand why optimizers are needed for neural network training.

Create an optimizer using torch.optim.

Use SGD in PyTorch.

Understand the basic idea of Adam.

Use optimizer.zero_grad().

Use optimizer.step().

Understand how loss, gradients, and optimizers work together.

Explain the basic PyTorch training cycle.

Lesson 8 — Training Loops

File: training_loop.py

Objectives

Understand what a training loop is.

Learn how to train a neural network for multiple epochs.

Understand the relationship between forward pass, loss, backward pass, and optimizer.

Learn how to track loss during training.

Understand the difference between a training step and an epoch.

Topics Covered

Training loops

Epochs

Forward pass

Loss calculation

Backward pass

Optimizer updates

Tracking training loss

optimizer.zero_grad()

loss.backward()

optimizer.step()

What is a Training Loop?

A training loop is the repeated process used to train a neural network.

The basic pattern is:

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
Parameter Update
  ↓
Repeat

A neural network usually needs many training steps to learn useful patterns.

What is an Epoch?

An epoch means one complete pass through the training data.

For example, if a model is trained for:

epochs = 10

the model goes through the training data 10 times.

Training Data
     ↓
Epoch 1
     ↓
Epoch 2
     ↓
Epoch 3
     ↓
...
     ↓
Epoch 10

Training Step vs Epoch

A training step updates the model parameters once using a batch of data.

An epoch contains all the training steps needed to process the complete training dataset once.

Epoch
 ├── Step 1
 ├── Step 2
 ├── Step 3
 └── Step 4

Example

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


x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
target = torch.tensor([[2.0], [4.0], [6.0], [8.0]])


epochs = 100

for epoch in range(epochs):

    optimizer.zero_grad()

    prediction = model(x)

    loss = loss_function(prediction, target)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            "Epoch:",
            epoch + 1,
            "Loss:",
            loss.item()
        )

What Happens in This Example?

Step 1 — Create the model

model = SimpleModel()

The model contains one linear layer.

Step 2 — Define the loss function

loss_function = nn.MSELoss()

MSE measures the difference between the model's prediction and the target.

Step 3 — Create the optimizer

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

The optimizer updates the model parameters.

Step 4 — Define the training data

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
target = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

The model is learning the relationship:

target = 2 × input

Step 5 — Set the number of epochs

epochs = 100

The model will process the training data 100 times.

Step 6 — Start the training loop

for epoch in range(epochs):

This repeats the training process for every epoch.

Step 7 — Clear gradients

optimizer.zero_grad()

This clears gradients from the previous training step.

Step 8 — Forward pass

prediction = model(x)

The input data passes through the model and produces predictions.

Step 9 — Calculate loss

loss = loss_function(prediction, target)

The loss tells us how different the predictions are from the targets.

Step 10 — Backward pass

loss.backward()

PyTorch calculates gradients for the model parameters.

Step 11 — Update parameters

optimizer.step()

The optimizer uses the gradients to update the model parameters.

Complete Training Pattern

The most important pattern to remember is:

for epoch in range(epochs):

    optimizer.zero_grad()

    prediction = model(x)

    loss = loss_function(prediction, target)

    loss.backward()

    optimizer.step()

This is one of the most important patterns in PyTorch.

Tracking Loss

During training, we usually monitor the loss.

print("Loss:", loss.item())

loss.item() converts the loss tensor containing a single value into a regular Python number.

We can also print the loss every few epochs:

if (epoch + 1) % 10 == 0:
    print("Epoch:", epoch + 1, "Loss:", loss.item())

If training is working correctly, the loss will generally decrease over time, although the exact behavior depends on the model, data, optimizer, and learning rate.

Training Flow

Training Data
      ↓
   Model
      ↓
 Prediction
      ↓
 Loss Function
      ↓
     Loss
      ↓
 backward()
      ↓
  Gradients
      ↓
optimizer.step()
      ↓
Updated Parameters
      ↓
     Repeat

Key Concepts to Remember

Training Loop

Repeats the model training process.

Epoch

One complete pass through the training data.

optimizer.zero_grad()

Clears old gradients.

Forward Pass

prediction = model(x)

Produces predictions from the input.

Loss

loss = loss_function(prediction, target)

Measures prediction error.

Backward Pass

loss.backward()

Calculates gradients.

Parameter Update

optimizer.step()

Updates the model parameters.

Learning Outcomes

After completing this lesson, I can:

Explain what a training loop is.

Explain what an epoch means.

Understand the difference between a training step and an epoch.

Create a basic PyTorch training loop.

Perform a forward pass.

Calculate loss.

Calculate gradients using backward().

Update model parameters using an optimizer.

Track training loss.

Explain the complete basic training process.

Overall Learning Progress

Lesson

Topic

Status

1

Tensors

Completed

2

Autograd

Completed

3

Gradient Descent

Completed

4

Neural Networks and nn.Module

Completed

5

Activation Functions

Completed

6

Loss Functions

Completed

7

Optimizers

Completed

8

Training Loops

Completed

Progress: 8/8 lessons completed

Concepts Learned So Far

The first eight lessons build the foundation for training PyTorch models.

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
Training Loop
   ↓
Repeated Parameter Updates

So far, I have learned how PyTorch represents data using tensors, calculates gradients using Autograd, updates parameters through gradient descent, creates neural networks using nn.Module, introduces non-linearity using activation functions, measures prediction error using loss functions, uses optimizers to update model parameters, and combines these concepts into a complete training loop.

Repository Structure

pytorch-learning/
│
├── README.md
├── tensors.py
├── autograd.py
├── gradient_descent.py
├── neural_network.py
├── activation_functions.py
├── loss_functions.py
├── optimizers.py
└── training_loop.py

Next Topics

The upcoming lessons will focus on:

Dataset and DataLoader

Batches

Training with datasets

Model evaluation

Training and validation

Saving and loading models

Building practical deep learning models
