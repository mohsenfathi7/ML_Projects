import torch.utils
import torch.nn as nn
import torch.cuda
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
root = './data'
download = True
trans = transforms.Compose([transforms.ToTensor()])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

batch_size = 100
kwargs = {'num_workers': 1}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=1,
                shuffle=False, **kwargs)

print( '==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


                    ####create MLPnet
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet,self).__init__()
        self.fc1=nn.Linear(28*28,500,bias=True)
        self.fc2=nn.Linear(500,256,bias=True)
        self.fc3 = nn.Linear(256, 10,bias=True)

        #####function neededd

    

    def RMSE(self,x, y):
        a = torch.dist(x, y, 2)
        return a

    def forward(self,x,target):
        x=x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x=self.fc3(x)
        loss=self.RMSE(x,target)
        return x,loss
    def name(self):
        return 'MLPnet'

model=MLPNet()#.cuda()
print(model)

epoch=10
learning_rate=0.15

optimizer=optim.SGD(model.parameters(),lr=learning_rate)
x_plot=np.zeros([epoch,1])
y_plot=np.zeros([epoch,1])

for k in range(epoch):
    #training
    avg_loss=0
    for  x,target in train_loader:
        optimizer.zero_grad()
        t = torch.zeros((batch_size, 10))
        model.train(True)
        x, target = x.type(torch.FloatTensor), target.type(torch.FloatTensor)
        x, target = Variable(x), Variable(target)
        for i in range(0,batch_size):
            tt=target.data.numpy()
            t[i,tt[i]]=1

        t=Variable(t)
        _, loss = model(x, t)
        avg_loss += loss.data[0] / len(train_loader)/batch_size
        loss.backward()
        optimizer.step()


#testing
    correct_cnt, avg_loss_test = 0, 0
    for x,target in test_loader:
        x, target = x.type(torch.FloatTensor), target.type(torch.FloatTensor)
        x, target = Variable(x,volatile=True), Variable(target,volatile=True)
        t = torch.zeros([1, 10])
        tt = target.data.numpy()
        t[0, tt[0]] = 1
        t = Variable(t)
        _,loss = model(x,t)
        
        avg_loss_test += loss.data[0] / len(test_loader)
    x_plot[k,0]=avg_loss
    y_plot[k,0]=avg_loss_test

plt.plot(x_plot,'-g')
plt.plot(y_plot,'b')
plt.show()
