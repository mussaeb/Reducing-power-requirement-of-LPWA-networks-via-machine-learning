import argparse
import torch 
from torch.autograd import Variable 
from torch.nn import functional as F 
from torch.utils.data import DataLoader
import sys
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(precision=1)


from models import *
from dataset import PowerDataset

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=200, help="total num of epochs")
parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="SGD: learning rate")
parser.add_argument("--input_size", type=int, default=15, help="number of input features")
parser.add_argument("--output_size", type=int, default=10, help="number of output features")
parser.add_argument("--model_type", type=str, default="deepnn", help="type of model to use")
parser.add_argument("--data_file", type=str, default='Data_set_no_formula.xlsx', help="type of model to use")
opt = parser.parse_args()

if opt.model_type == 'svr':
	model = SupportVectorRegression(opt.input_size, opt.output_size)
elif opt.model_type == 'deepnn':
	model = DeepNN(hidden_size=100, input_size=opt.input_size, output_size=opt.output_size)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)

model = model.float() 

# Training data loader
dataloader = DataLoader(
    PowerDataset(opt.data_file),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Test data loader
val_dataloader = DataLoader(
	PowerDataset(path=opt.data_file, mode="test"),
    batch_size=5,
    shuffle=True,
    # num_workers=1,
)


#### Trainging ####
running_loss = 0.0
for epoch in range(opt.epoch):
	for i, batch in enumerate(dataloader):
		y_val = batch['y_val']
		x_val = batch['x_val']
		optimizer.zero_grad()

		y_pred = model(x_val.float())

		# loss = criterion(y_pred, y_val)
		loss = criterion(y_pred.float(), y_val.float())
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		if i%5 ==4:
			sys.stdout.write(
				"\r[Epoch %d] [Batch %d] [MSELoss loss: %f]"
			    % (
			        epoch,
			        i,
			        loss.item(),
			    )
			)
			running_loss = 0.0
print('\nFinished Training')


#### Testing model and printing absolute difference between real and predicted #### 
length_data = len(val_dataloader)
print(length_data)
model.eval()
for i, batch in enumerate(val_dataloader):
	y_val_test = batch['y_val']
	x_val_test = batch['x_val']

	y_pred_test = model(x_val_test.float())
	# print("Real Values: \t",  y_val_test)
	# print("Predicted Values: \t",  y_val_test)
	if i == 0: 
		result = (abs(y_pred_test - y_val_test)).data.numpy()
		break 
	#print("Absolute Difference:\t", (abs(y_pred_test - y_val_test)).data.numpy())
print("predicted", y_pred_test[-1].data.numpy())
print("real_value", y_val_test[-1].data.numpy())

# print(result.shape)
# new_xticks = ['Tsym ms', 
# 			'PayloadSymbNb ms', 
# 			'Tpacket ms', 
# 			'TTN 30sec/Day Fair Use', 
# 			"Radio/MPU(per cycle period)",	
# 			"lag/Lead MCU (per Cycle Period)",	
# 			"Sensor (Per Cycle period)",	
# 			"Sleep Current (in Cycle period)",	
# 			"Summary (per Tx Cycle)", 
# 			"Avg Battery life in (Years)"]

plt.title("Absolute Difference of Read and Predicted")
plt.xlabel("Predicted Colums (tsym etc)")
plt.ylabel("Absolute Difference")
for data in result: 
	plt.plot(data)
plt.show()
