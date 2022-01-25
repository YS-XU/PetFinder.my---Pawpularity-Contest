from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import math
import pickle

# The parts that you should complete are designated as TODO
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO: define the layers of the network
        self.cnn = nn.Sequential(nn.Conv2d(3, 32, (5, 5), stride = 1), nn.ReLU(), 
                                nn.MaxPool2d(2, stride = 2), 

                                nn.Conv2d(32, 32, (3, 3), stride = 1, padding=1), nn.ReLU(), 
                                nn.BatchNorm2d(32),
                                nn.MaxPool2d(2, stride = 2), 
                                
                                nn.Conv2d(32, 64, (3, 3), stride = 1, padding=1), nn.ReLU(),  
                                # nn.Conv2d(64, 64, (3, 3), stride = 1, padding=1), nn.ReLU(), 
                                nn.BatchNorm2d(64),
                                nn.MaxPool2d(2, stride = 2), 

                                nn.Conv2d(64, 128, (3, 3), stride = 1, padding=1), nn.ReLU(), 
                                # nn.Conv2d(128, 128, (3, 3), stride = 1, padding=1), nn.ReLU(), 
                                nn.BatchNorm2d(128),
                                nn.MaxPool2d(2, stride = 2), 

                                nn.Conv2d(128, 256, (3, 3), stride = 1, padding=1), nn.ReLU(), 
                                nn.BatchNorm2d(256),
                                nn.MaxPool2d(2, stride = 2), 
                                # nn.Dropout2d(p = 0.25), 

                                nn.Conv2d(256, 512, (3, 3), stride = 1, padding=1), nn.ReLU(), 
                                nn.BatchNorm2d(512),
                                nn.MaxPool2d(2, stride = 2), 

                                nn.Conv2d(512, 1024, (3, 3), stride=1), 
                                nn.BatchNorm2d(1024),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 256), 
            nn.ReLU(),
            # nn.Dropout(p = 0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        out = self.cnn(x)
        # print(out.shape)
        # exit()
        return self.fc(out)

def train(model, device, train_loader, optimizer, loss_func, epoch, train_y):
    model.train()
    index = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).float(), target.to(device)
        #criterion = F.cross_entropy(output, target)
        optimizer.zero_grad()
        output = model(data).squeeze()
        # pred = output.argmax(dim=1, keepdim=True)
        loss = torch.sqrt(loss_func(output, target))
        # Backward pass
        loss.backward()
        optimizer.step()
        accuracy = 100. * batch_idx / len(train_loader)
        if batch_idx % 123 == 0: #Print loss every 100 batch
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                accuracy, loss.item()))
    return model

def test(model, device, test_loader, loss_func, test_y):
    model.eval()
    correct = 0
    predict = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device)
            output = model(data)
            pred = output
            predict.append(pred.item())
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    
    predict = torch.tensor(predict, dtype=torch.float, device=device)
    rmse = torch.sqrt(loss_func(predict, test_y)).item()
    print("Test RMSE: ", rmse)
    
    return rmse

def load_data():
    """
    """
    width = 256
    heigh = 256
    data_cov = pd.read_csv(f"../petfinder-pawpularity-score/train.csv", header=None).to_numpy()
    data_X = data_cov[:, 0]
    data_X = np.delete(data_X, 0, 0)
    data_y = data_cov[:, len(data_cov[0]) - 1]
    data_y = np.delete(data_y, 0, 0)
    data_y = data_y.astype(dtype=int)
    print(data_y)
    data = []
    image_width = []
    image_length = []
    
    # for i in range(len(data_X)):
    #     image = Image.open(f"../petfinder-pawpularity-score/train/{data_X[i]}.jpg")
    #     image = image.resize((width, heigh))
    #     data.append(np.array(image).T)
    # data = np.array(data)
    # print(data.shape)
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data.shape)
    print(data_y.shape)
    

    return data, data_y

def main():
    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = False # Switch to False if you only want to use your CPU
    learning_rate = 1e-2 # best 1e-2
    NumEpochs = 7
    batch_size = 32 #best 32

    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_X, train_Y = load_data()
    print(train_X.shape)
    print(train_Y.shape)
    
    tensor_x = torch.tensor(train_X, device=device) 
    tensor_y = torch.tensor(train_Y, dtype=torch.float, device=device)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    
    model = ConvNet().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=learning_rate,

    )
    loss_func = torch.nn.MSELoss()


    for epoch in range(NumEpochs):
        final_model = train(model, device, train_loader, optimizer, loss_func, epoch, train_Y)
    with open('model2.pkl', 'wb') as f:
        pickle.dump(final_model, f)



if __name__ == '__main__':
    main()