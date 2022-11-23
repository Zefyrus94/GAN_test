import torch
import torch.nn as nn
import torch.optim as optim

print("model parallel example")
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:0')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:1')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:0')))
        return self.net2(x.to('cuda:1'))

######################################################################
# Note that, the above ``ToyModel`` looks very similar to how one would
# implement it on a single GPU, except the four ``to(device)`` calls which
# place linear layers and tensors on proper devices. That is the only place in
# the model that requires changes. The ``backward()`` and ``torch.optim`` will
# automatically take care of gradients as if the model is on one GPU. You only
# need to make sure that the labels are on the same device as the outputs when
# calling the loss function.


model = ToyModel()
from torchsummary import summary
summary(model, (20,10))

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1')
loss_fn(outputs, labels).backward()
optimizer.step()