import torch

# Create a tensor
x = torch.tensor(3.0, requires_grad=True)

# Perform a calculation
y = x ** 2

# Calculate the gradient
y.backward()

# Print the gradient
print("x =", x)
print("y =", y)
print("Gradient =", x.grad)
