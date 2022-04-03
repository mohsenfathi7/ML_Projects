import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
        #####prepare dataset for train and test
root = './data'
download = True
trans = transforms.Compose([transforms.ToTensor()])
train_set = dset.MNIST(root=root, train=True,transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False,transform=trans)


batch_size = 100
kwargs = {'num_workers': 1}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False, **kwargs)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.fc1 = nn.Linear(28*28, 150,bias=True)
        self.fc2 = nn.Linear(150, 28*28,bias=True)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.sigmoid(self.fc1(x))
        encode=x
        x = F.sigmoid(self.fc2(x))
        return x,encode
    def name(self):
        return 'mlpnet'

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet,self).__init__()
        self.fc1=nn.Linear(150,50,bias=True)
        self.fc2=nn.Linear(50,10,bias=True)
        #####function neededd

   

    def RMSE(self,x, y):
        a = torch.dist(x, y, 2)
        return a

    def forward(self,x,target):
        x=x.view(-1,150)
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        loss=self.RMSE(x,target)
        return x,loss
    def name(self):
        return 'MLPnet'

model_mpl=MLPNet()#.cuda()

model_auto = autoencoder()
print(model_mpl)
print(model_auto)
learning_rate = 0.01
optimizer = optim.SGD(model_auto.parameters(), lr=learning_rate,momentum=0.9)
# optimizer = optim.SGD(model_mpl.parameters(), lr=learning_rate)

#ceriation = nn.CrossEntropyLoss()
ceriation = nn.MSELoss()
epoch = 15
x_plt=np.zeros([epoch,1])
y_plt=np.zeros([epoch,1])
for epoch in range(epoch):
    avg_loss = 0
    train_accuracy = 0
    # Training
    for x, target in train_loader:
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)

        _,out = model_auto(x)
        h=torch.zeros([batch_size,10])
        for i in range(batch_size):
            tt=target.data.numpy()
            h[i,tt[i]]=1

        h=Variable(h)
        _,loss = model_mpl(out, h)
        loss.backward()
        optimizer.step()
        avg_loss += loss.data[0]/len(train_loader)/batch_size
    x_plt[epoch,0]=avg_loss
    print("Epoch:", epoch+1, "Train Loss:", avg_loss)
    test_loss = 0
    for x, target in test_loader:
        x, target = Variable(x, volatile=True),Variable(target, volatile=True)
        _,out= model_auto(x)
        h = torch.zeros([batch_size, 10])
        for i in range(batch_size):
            tt = target.data.numpy()
            h[i, tt[i]] = 1

        h = Variable(h)
        _,loss = model_mpl(out, h)
        test_loss+= loss.data[0] / len(test_loader)/batch_size
    y_plt[epoch,0]= test_loss;

    print("loss_test", test_loss)


plt.plot(x_plt,'r-')
plt.plot(y_plt,'b')
plt.show()

