```python
import torch

# PyTorch Lesson 3 - Gradient Descent

# Create a parameter that we want to optimize
x = torch.tensor(5.0, requires_grad=True)

# Learning rate
learning_rate = 0.1

print("Initial x =", x.item())

# Gradient descent for several steps
for step in range(10):

    # Calculate the function
    # We want x to become 0
    y = x ** 2

    # Calculate the gradient
    y.backward()

    # Update x using the gradient
    with torch.no_grad():
        x -= learning_rate * x.grad

    # Clear the old gradient
    x.grad.zero_()

    print("Step", step + 1, "x =", x.item(), "y =", y.item())

print("\nFinal x =", x.item())
```
