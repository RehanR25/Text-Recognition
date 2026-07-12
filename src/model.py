from torch import nn 

import torch.nn.functional as F

class model(nn.Module): 
    def __init__(self, in_channels, num_classes):
        super(model, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        # in_channels = in_channels that way can do both RGB and grayscale  

        # Pooling layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # 3rd convolutional layer
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # 4th convolutional layer
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        """
        The convolutional layers are stacked together so the 2nd layer can refine what the 3rd 
        layer already extracted at the same channel depth. This adds more depth (more non linear transformations)
        without increasing the channel count further keeping compute costs low and allowes network to build more 
        complex features from the same dimensional feature space

        """
        # Pooling layer 
        self.pool2 = nn.MaxPool2d(kernel_size=(2,1))
        # Assymmetric pooling since width allows for sequence information unlike height

        # 5th convolutional layer
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        # Batch Normalisation 
        self.bn = nn.BatchNorm2d(512)

        # 6th convolutional layer 
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # 7th convolutional layer
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding=0)

        # 1st recurrent layer
        self.rnn1 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True)

        # 2nd recurrent layer
        self.rnn2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True)

        self.fc = nn.Linear(512,num_classes)


    def forward(self,x) : 
        """
        Defining the forward pass of the neural network 

        Parameters : 
            x : Input tensor 
        Returns : 
            torch.tensor (output tnesor after passing through network)
        """

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.bn(self.conv5(x)))
        x = F.relu(self.bn(self.conv6(x)))
        x = self.pool2(x)
        x = F.relu(self.conv7(x))

        x = x.squeeze(2)     # removes the height from the CNN output
        x = x.permute(2,0,1) # reorders dimensions without changing the data itself 
                             # such that (batch, 512, width) becomes (width, batch, 512)
                             # since nn.LTSM expects (sequence_length, batch, features), not (batch, features, sequence_length)
        x, (hidden, cell) = self.rnn1(x) 
        x, (hidden, cell) = self.rnn2(x)
        x = self.fc(x)       # applies a fully connected layer

        return x 


# Use nn.Sequential later on 
# See if map_to_seq is worth it or not to shrink CNN output
