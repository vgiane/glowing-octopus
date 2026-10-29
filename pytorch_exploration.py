import torch
from mpmath import fac
from torch import nn, optim
from torchvision.models import ResNet18_Weights, resnet18

# %%
shape = (2, 3, 2)
x_rand_data = torch.rand(shape, dtype=torch.float32)
x_ones_data = torch.ones(shape, dtype=torch.float32)
x_zeros_data = torch.zeros(shape, dtype=torch.float32)

print(x_rand_data)

# We move our tensor to the GPU if available
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
tensor = x_rand_data.to(device)
print(f"Device tensor is stored on: {tensor.device}")
# %%
y = torch.tensor([[[1, 2.0], [3, 4], [5, 6]], [[1, 2], [3, 4], [5, 6]]])
y.copy_(x_rand_data)
print(y)

# %%
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
output = model(data)

loss = (output - labels).sum()
loss.backward()

# %%
first_optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
first_optim.step()

# At this point, you have everything you need to train your neural network.
# %%
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)
print(a, b)

# %%
Q = 3 * a**2 - b**2

# %%
external_grad = torch.tensor([1.0, 1.0])
Q.backward(gradient=external_grad)
print(a.grad, b.grad)

# %%
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, 10)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
