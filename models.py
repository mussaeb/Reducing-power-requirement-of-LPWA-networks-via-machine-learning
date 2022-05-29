import torch 
from torch.nn import functional as F 


class LinearRegression(torch.nn.Module):
	def __init__(self, input_size):
		super(LinearRegression, self).__init__()
		self.linear = torch.nn.Linear(input_size,1)

	def forward(self,x):
		y_pred = self.linear(x)
		return y_pred

class SupportVectorRegression(torch.nn.Module):
	def __init__(self, input_size, output_size):
		super(SupportVectorRegression, self).__init__()
		self.linear = torch.nn.Linear(input_size,output_size)

	def forward(self,x):
		y_pred = self.linear(x)
		return y_pred

class DeepNN(torch.nn.Module):
	def __init__(self,hidden_size, input_size, output_size):
		super(DeepNN, self).__init__()
		self.fc1 = torch.nn.Linear(input_size,hidden_size)
		self.relu1 = torch.nn.ReLU()
		self.fc2 = torch.nn.Linear(hidden_size,hidden_size)
		self.relu2 = torch.nn.ReLU()
		self.fc3 = torch.nn.Linear(hidden_size,output_size)

	def forward(self,x):
		out = self.fc1(x)
		out = self.relu1(out)
		out = self.fc2(out)
		out = self.relu2(out)
		out = self.fc3(out)
		return out

