import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
dtype = torch.cuda.FloatTensor

# XOR Problems

class dnn(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 2
        Channel_0 = 10
        Channel_1 = 10
        output = 1
        # self.weight0 = torch.cuda.FloatTensor(input_size, Channel_0)
        self.fc1 = nn.Linear(input_size,Channel_0)
        self.fc2 = nn.Linear(Channel_0, Channel_1)
        self.fc3 = nn.Linear(Channel_1,1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def make_data(length, dtype):
    data = np.random.rand(length, 2)

    np.random.shuffle(data)
    target = np.zeros((length, 1), dtype=np.float16)
    for i in range(length):
        if data[i, 0] >= 0.5 and data[i, 1] >= 0.5:
            target[i] = 0
        elif data[i, 0] < 0.5 and data[i, 1] >= 0.5:
            target[i] = 1
        elif data[i, 0] >= 0.5 and data[i, 1] < 0.5:
            target[i] = 1
        elif data[i, 0] < 0.5 and data[i, 1] < 0.5:
            target[i] = 0
    return torch.from_numpy(data).type(dtype), torch.from_numpy(target).type(dtype)


my_dnn = dnn()
my_dnn = my_dnn.cuda()
optimizer = torch.optim.Adam(my_dnn.parameters(), lr=0.01)

data,target = make_data(5000, dtype)
test,test_target = make_data(50, dtype)

test_for_x1 =torch.from_numpy(np.linspace(0,1,100,dtype = np.float16)).type(dtype)
test_for_x2 = torch.from_numpy(np.linspace(0,1,100,dtype = np.float16)).type(dtype)
test_for_y = np.zeros((100,100,1),dtype=np.float16)
# print(test_for_vis.shape)
def train(epoch):
    my_dnn.train()
    for ii in range(100):
        data_tensor = data[ii*50:ii*50+50,:]

        target_tensor = target[ii*50:ii*50+50]
        optimizer.zero_grad()
        output = my_dnn(data_tensor)
        loss = F.l1_loss(output,target_tensor)
        loss.backward()
        optimizer.step()
        # print(loss)

        # print('Train Epoch: {} [ {}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(epoch, (ii+1)*50, 1000 ,100.*ii/1000),loss(0))
#     for
# my_dnn(data)
for epoch in range(1, 10+1):
    train(epoch)
    # print(my_dnn(test))


# out = my_dnn(test)>=0.5
# print((out.type(dtype) - test_target == 0.).sum().type(dtype)/50*100)
for i in range(100):
    for j in range(100):
        test_for_y[i,j,0] = my_dnn(torch.FloatTensor([test_for_x1[i],test_for_x2[j]]).type(dtype)).data.cpu().numpy()
        print(test_for_y[i,j,0])

# print(test_for_y[i,j,0])
test_for_y = test_for_y+test_for_y.min()
test_for_y = test_for_y/test_for_y.max()
test_for_y = test_for_y*255
test_for_y = test_for_y.astype(np.uint8)
cv2.imshow('win',test_for_y)
cv2.waitKey(0)