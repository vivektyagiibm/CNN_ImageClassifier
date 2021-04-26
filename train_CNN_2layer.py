import numpy as np 
import torch 
import numpy as np
from collections import defaultdict, OrderedDict
import re
import warnings
import sys
import time
#import _pickle as cPickle
import pickle as cPickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import autograd

import gc

# very important to set np random seed for reproducibility
np.random.seed(0)
torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# First checking if GPU is available
train_on_gpu=torch.cuda.is_available()



torch.set_printoptions(profile="full")
torch.set_printoptions(precision=3)



import utils.mnist_reader


X_train, y_train = utils.mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = utils.mnist_reader.load_mnist('data/fashion', kind='t10k')


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class CNN(nn.Module):
    def __init__(self, C, Ci, Co):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(Ci, Co, (3,3))
        self.conv2 = nn.Conv2d(Co, Co, (3,3))
        self.fc1 = nn.Linear(32*5*5, C)
    
    def forward(self, x):
        batch_size = x.shape[0]
        #print(x.shape)    
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x= F.max_pool2d(x, (2,2))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x= F.max_pool2d(x, (2,2))
        #print(x.shape)
        x = x.view(batch_size,-1)
        #print(x.shape)
        x = self.fc1(x)
        return (x)

n_epochs = 50
batch_size = 20
C = 10
np.random.seed(123)
lrnR = 0.008
n_train_batches = int(X_train.shape[0]/batch_size)



cnn =CNN(10, 1, 32)
cnn = cnn.to(device)
print(cnn)
pytorch_total_params = sum(p.numel() for p in cnn.parameters())
print('total number of params:'+str( pytorch_total_params))

epoch =0
steps = 0
criterion = nn.CrossEntropyLoss()
while (epoch < n_epochs):
    cnn.train()
    start_time = time.time()
    epoch = epoch + 1
    for index in np.random.permutation(range(n_train_batches)):
    #for index in (range(1)):
        x = X_train[index*batch_size:(index+1)*batch_size]
        y = y_train[index*batch_size:(index+1)*batch_size]
        x = x.reshape(batch_size, 1, 28,28)
        xTensor = torch.from_numpy(x).float()
        yTensor = torch.from_numpy(y).long()

        xTensor = xTensor.to(device)
        yTensor = yTensor.to(device)
        cnn.zero_grad()
        logit = cnn(xTensor)
        loss = criterion(logit,yTensor)
        loss.backward()


        for name, param in cnn.named_parameters():
                param.data.add_(-lrnR, param.grad.data)
        steps += 1
        if steps % 500 == 0:
            print("steps:"+str(steps)+' loss='+str(loss))
            corrects = (torch.max(logit, 1)[1].view(yTensor.size()).data == yTensor.data).sum()
            accuracy = 100.0 * corrects/batch_size
            print("Epoch="+str(epoch)+"steps="+str(steps)+ " accuracy="+str(accuracy) )
            print('index='+str(index)+' loss='+str(loss))



x = X_test.reshape(-1, 1,28,28)
y = y_test
xTensor = torch.from_numpy(x).float()
yTensor = torch.from_numpy(y).long()
xTensor = xTensor.to(device)
yTensor = yTensor.to(device)

logit = cnn(xTensor)
corrects = (torch.max(logit, 1)[1].view(yTensor.size()).data == yTensor.data).sum()
accuracy = 100.0 * corrects/y.shape[0]
print('accuracy:'+str(accuracy))
print('corrrects:'+str( corrects))








