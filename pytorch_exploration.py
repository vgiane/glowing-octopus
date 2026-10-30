import einops
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn.functional import relu
from torchvision.transforms import v2


# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(3072, 256)
        self.f2 = nn.Linear(256, 10)

    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b (c h w)")
        x = relu(self.f1(x))
        x = self.f2(x)
        return x


criterion = nn.CrossEntropyLoss()


# %%
def load_data(train):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    batches = torchvision.datasets.CIFAR10(
        root="./data", train=train, download=False, transform=transform
    )
    return batches


batches = torch.utils.data.DataLoader(
    load_data(train=True), batch_size=64, shuffle=True, num_workers=0
)


def check_initial_loss(net, x, y):
    output = net(x)
    loss = criterion(output, y)
    return loss.item()


first_batch = next(iter(batches))

assert Net()(first_batch[0]).shape == (64, 10)

assert np.isclose(
    check_initial_loss(Net(), first_batch[0], first_batch[1]), 2.3, atol=0.2
)


def check_overfit(net, x_batch, y_batch):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for _ in range(200):
        optimizer.zero_grad()
        output = net(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    return criterion(net(x_batch), y_batch).item()


assert np.isclose(check_overfit(Net(), first_batch[0], first_batch[1]), 0.0, atol=0.05)

# %%

def training_loop(net, batches, epochs=2):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(epochs):
        for i, (x_batch, y_batch) in enumerate(batches):
            optimizer.zero_grad()
            output = net(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch}, batch {i}, loss {loss.item()}")
    return net


net = training_loop(Net(), batches)

# %% test the net on the test set
testset = load_data(train=False)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=0
)
net.eval()
correct = 0
with torch.no_grad():
    for x, y in testloader:
        prediction = net(x).argmax(dim=1)
        correct += (prediction == y).sum().item()
    print(f"Accuracy: {correct / len(testset)}")
