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
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from PIL import Image
from numpy.linalg import inv
from scipy.spatial.distance import cdist


        #####prepare dataset for train and test
root = './data'
download = True
trans = transforms.Compose([transforms.ToTensor()])
train_set = dset.MNIST(root=root, train=True, transform=trans, download=download)
test_set = dset.MNIST(root=root, train=False, transform=trans)

pca0=PCA(n_components=150,svd_solver='full')
z=train_set.train_data
final0=pca0.fit_transform(np.array(z).reshape(60000,784))
train_set=torch.utils.data.TensorDataset(torch.from_numpy(final0),train_set.train_labels)

pca1=PCA(n_components=150,svd_solver='full')
zz=test_set.test_data
print(zz.size())
final1=pca1.fit_transform(np.array(zz).reshape(10000,784))
test_set=torch.utils.data.TensorDataset(torch.from_numpy(final1),test_set.test_labels)

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

print( '==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))


                    ####create MLPnet
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

model=MLPNet()#.cuda()
print(model)

epoch=15
learning_rate=0.01

optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9)
x_plot=np.zeros([epoch,1])
y_plot=np.zeros([epoch,1])
imm = transforms.Compose([transforms.ToPILImage()])

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

    print(' #epoch# ',k, 'train_loss', avg_loss)
#testing
    a=np.floor(np.random.rand(1)*len(test_loader))
    d=0
    avg_loss_test = 0
    for x,target in test_loader:
        d=d+1
        x, target = x.type(torch.FloatTensor), target.type(torch.FloatTensor)
        x, target = Variable(x,volatile=True), Variable(target,volatile=True)
        t = torch.zeros([batch_size, 10])
        tt = target.data.numpy()
        for i in range(batch_size):
            tt = target.data.numpy()
            t[i, tt[i]] = 1

        t = Variable(t)
        _,loss = model(x,t)

        avg_loss_test += loss.data[0] / len(test_loader)/batch_size
    print('test_loss',avg_loss_test)
    x_plot[k,0]=avg_loss
    y_plot[k,0]=avg_loss_test

plt.plot(x_plot,'-g')
plt.plot(y_plot,'b')
plt.show()
