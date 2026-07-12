import matplotlib.pyplot as plt 
import numpy as np 

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm 
from torch import optim

import torchvision
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 

from torchmetrics import Accuracy, Precision, Recall


train_dataset = datasets.MNIST(root="dataset/", download=True, train=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size= 60, shuffle=True)

test_dataset = datasets.MNIST(root="dataset/", download=True, train=False, transform=transforms.ToTensor())

test_loader = DataLoader(dataset=test_dataset, batch_size=60, shuffle=True) 

def imshow(img): 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(train_loader)
images, _ = next(dataiter)

imshow(torchvision.utils.make_grid(images))



class model1(nn.Module): 
    """
    Building blocks of the convolutional neural network 

    Parameters : 
        * in_channels : Number of channels in the input image 
        * num_classes : Number of classes to predict. In this problem 10(number of digits - 0 to 9)
    """
    def __init__(self, in_channels, num_classes):
        super(model1, self). __init__()
        
        # 1st convolutional layer 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        # Pooling Layer 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 2nd convolutional layer 
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x) : 
        
        """
        Define the forward pass of the neural network 

        Parameters : 
            x : Input tensor 
        Returns : 
            torch.Tensor (output tensor after passing through the network)
        """

        # Applies the ReLU activation on the first convolution
        x = F.relu(self.conv1(x))
        # Apply max pooling 
        x = self.pool(x)
        # Applies the ReLU activation on the sencond convolution
        x = F.relu(self.conv2(x))
        # Apply the max pooling 
        x = self.pool(x)
        # Flatten the tensor
        x = x.reshape(x.shape[0], -1) 
        # Apply fully connected layer 
        x = self.fc1(x)
        return x  
    

device = "cuda" if torch.cuda.is_available() else "cpu"

cnn_model = model1(in_channels=1, num_classes=10).to(device)

## Training 

#Define the loss function 
criterion = nn.CrossEntropyLoss()

# Define the optimiser 
optimiser = optim.Adam(cnn_model.parameters(), lr=0.001)

num_epochs = 10 
for epoch in range(num_epochs): 
    # Iterate over training batches 
    print(f"Epoch [{epoch+1}/{num_epochs}]")

    for batch_index, (data, target) in enumerate(tqdm(train_loader)): 
        data = data.to(device)
        target = target.to(device)
        scores = cnn_model(data)
        loss = criterion(scores,target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

acc= Accuracy(task="multiclass", num_classes=10).to(device)
precision = Precision(task="multiclass", num_classes=10).to(device)
recall = Recall(task="multiclass", num_classes=10).to(device)

# Iterate over the dataset batches 
cnn_model.eval() 
with torch.no_grad(): 
    for images, labels in test_loader: 
        # Get predicted probabilities for test data batch 
        outputs = cnn_model(images)
        _, preds = torch.max(outputs, 1)
        acc(preds, labels)
        precision(preds, labels)
        recall(preds, labels)

# Compute total test accuracy
test_accuracy = acc.compute()
print(f"Test accuracy: {test_accuracy}")