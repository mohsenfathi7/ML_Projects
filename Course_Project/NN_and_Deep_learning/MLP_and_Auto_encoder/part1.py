import torch.utils
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import torch.utils.data.dataloader



        #####prepare dataset for train and test
train=np.random.rand(500,3)*5+1
t_train=np.zeros([500,1])
test=np.random.rand(200,3)*4+1.5
t_test=np.zeros([200,1])

for k in range(500):
    t_train[k,0]=(1+train[k,0]**0.5 + train[k,1]**-1 + train[k,2]**-1.5)**2
for k in range(200):
    t_test[k,0]=(1+test[k,0]**0.5 + test[k,1]**-1 + test[k,2]**-1.5)**2


trans=transforms.Compose([transforms.ToTensor()])

train=torch.from_numpy(train)   #####convert to torch.doubleTensor
t_train=torch.from_numpy(t_train)
train_set=torch.utils.data.TensorDataset(train,t_train)

test=torch.from_numpy(test)
t_test=torch.from_numpy(t_test)
test_set=torch.utils.data.TensorDataset(test,t_test)

train_loader=torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=1,
    shuffle=False)
test_loader=torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=1,
    shuffle=False)


                    ####create MLPnet
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet,self).__init__()
        self.fc1=nn.Linear(3,64,bias=True)
        self.fc2=nn.Linear(64,1,bias=True)

        #####function neededd

    

    def RMSE(self,x, y):
        a = torch.dist(x, y, 2)
        return a

    def forward(self,x,target):
        x = F.relu(self.fc1(x))
        x=self.fc2(x)
        loss=self.RMSE(x,target)
        return x,loss
    def name(self):
        return 'MLPnet'

model=MLPNet()#.cuda()

epoch=100
learning_rate=0.1

optimizer=optim.SGD(model.parameters(),lr=learning_rate)
x_plot=np.zeros([epoch,1])
y_plot=np.zeros([epoch,1])

for k in range(epoch):
    #training
    avg_loss=0
    for x,target in train_loader:
        optimizer.zero_grad()
        model.train(True)
        x, target = x.type(torch.FloatTensor), target.type(torch.FloatTensor)
        x, target = Variable(x), Variable(target)

        #out=model(x)
        _,loss=model(x,target)
        #loss=MAPE(out,target)
        #loss = MAPE(out, target)
        avg_loss += loss.data[0] / len(train_loader)
        loss.backward()
        optimizer.step()

#testing
    correct_cnt, avg_loss_test = 0, 0
    for x,target in test_loader:
        x, target = x.type(torch.FloatTensor), target.type(torch.FloatTensor)
        x, target = Variable(x), Variable(target)
        #out=model(x)
        _,loss = model(x,target)
       
        avg_loss_test += loss.data[0] / len(test_loader)
    x_plot[k,0]=avg_loss
    y_plot[k,0]=avg_loss_test

plt.plot(x_plot,'-g')
plt.plot(y_plot,'b')
plt.show()
