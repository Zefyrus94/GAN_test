import torch
import torch.nn as nn
import torch.optim as optim
#
import numpy as np
import timeit
#
print("model parallel example")
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10).to('cuda:2')
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5).to('cuda:3')

    def forward(self, x):
        x = self.relu(self.net1(x.to('cuda:2')))
        return self.net2(x.to('cuda:3'))

######################################################################
# Note that, the above ``ToyModel`` looks very similar to how one would
# implement it on a single GPU, except the four ``to(device)`` calls which
# place linear layers and tensors on proper devices. That is the only place in
# the model that requires changes. The ``backward()`` and ``torch.optim`` will
# automatically take care of gradients as if the model is on one GPU. You only
# need to make sure that the labels are on the same device as the outputs when
# calling the loss function.
#ToyModelSplit
class ToyModelSplit(ToyModel):
	def __init__(self, split_size=20, *args, **kwargs):
		super(ToyModelSplit, self).__init__(*args, **kwargs)
		self.split_size = split_size

	def forward(self, x):
		splits = iter(x.split(self.split_size, dim=0))
		s_next = next(splits)
		s_prev = self.relu(self.net1(x.to('cuda:2')))
		ret = []

		for s_next in splits:
		    # A. s_prev runs on cuda:1
		    s_prev = self.net2(x.to('cuda:3'))
		    ret.append(s_prev)

		    # B. s_next runs on cuda:2, which can run concurrently with A
		    s_prev = self.relu(self.net1(x.to('cuda:2')))

		s_prev = self.net2(s_prev.to('cuda:3'))
		ret.append(s_prev)

		return torch.cat(ret)

def train(model):
	loss_fn = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001)

	optimizer.zero_grad()
	outputs = model(torch.randn(20, 10))
	labels = torch.randn(20, 5).to('cuda:1')
	loss_fn(outputs, labels).backward()
	optimizer.step()

is_split = True
if not is_split:
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
else:
	print("split...")
	model = ToyModelSplit()
	print(model)
	print("params",model.parameters())
	train(model)
	"""
	stmt = "train(model)"
	setup = "model = ToyModelSplit()"
	num_repeat = 10
	pp_run_times = timeit.repeat(
	    stmt, setup, number=1, repeat=num_repeat, globals=globals())
	pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)
	print(pp_run_times,pp_mean,pp_std)
	"""