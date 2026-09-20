```python
import torch

# PyTorch Autograd
# Autograd automatically calculates gradients.

# Create a tensor and enable gradient tracking
x = torch.tensor(3.0, requires_grad=True)

# Forward calculation
y = x ** 2

print("x =", x)
print("y =", y)

# Calculate the gradient
y.backward()

# Print the gradient
print("Gradient of x =", x.grad)


# Example with multiple variables
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

# Forward calculation
y = w * x

print("\nMultiple Variables")
print("y =", y)

# Calculate gradients
y.backward()

print("Gradient of x =", x.grad)
print("Gradient of w =", w.grad)
```
