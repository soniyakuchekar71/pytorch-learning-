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
