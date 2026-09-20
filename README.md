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

## Overall Learning Progress

| Lesson | Topic            | Status    |
| ------ | ---------------- | --------- |
| 1      | Tensors          | Completed |
| 2      | Autograd         | Completed |
| 3      | Gradient Descent | Completed |

---

## Concepts Learned So Far

The first three lessons cover the basic process behind optimization in machine learning:

```text
Tensor
   ↓
Calculation
   ↓
Gradient
   ↓
Parameter Update
   ↓
Repeat
```

So far, I have learned how PyTorch represents data using tensors, calculates gradients using Autograd, and uses those gradients to update parameters through gradient descent.

---

## Repository Structure

```text
pytorch-learning/
│
├── README.md
├── tensors.py
├── autograd.py
└── gradient_descent.py
```

---

## Next Topics

The upcoming lessons will focus on:

* Neural network fundamentals
* `nn.Module`
* Loss functions
* Optimizers
* Model training
* Training and validation
* Building practical neural networks

---

## Goal

Build a strong foundation in PyTorch and gradually progress from basic tensor operations to implementing and training practical deep learning models.
