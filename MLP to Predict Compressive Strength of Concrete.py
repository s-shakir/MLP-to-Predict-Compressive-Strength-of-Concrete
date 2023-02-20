
# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive
drive.mount('/gdrive')
# %cd /gdrive

from google.colab import drive
drive.mount('/content/gdrive')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/gdrive/My Drive

"""Model Training"""

import numpy as np
import torch # tensor library
from torch import nn # neural network library
from torch.utils.data import DataLoader, TensorDataset # libraries for data loading and combining tensors
from sklearn.model_selection import KFold
# Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (without shuffling by default).
# Each fold is then used once as a validation while the k - 1 remaining folds form the training set.
import pandas as pd


class Net(nn.Module): # inheritance of base class for NN nn.module
    # Initialize the layers
    def __init__(self):
        super().__init__() # for calling constructor of nn.module
        self.linear1 = nn.Linear(8, 30) # hidden layer, input = 8, output = 30
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(30, 1) # output layer, input = 30, output = 1

    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


if __name__ == '__main__':

    # Configuration options
    k_folds = 5
    num_epochs = 15
    loss_funct = nn.MSELoss()

    # For fold results
    results = {}

    # Set fixed random number seed
    torch.manual_seed(42)


    Xtr=np.loadtxt("TrainData.csv")
    Ytr=np.loadtxt("TrainLabels.csv")

    # conversion from numpy to tensors
    Xdat = torch.from_numpy(Xtr).type(torch.float)
    Ydat = torch.from_numpy(Ytr).type(torch.float).unsqueeze(-1) # -1 to add dimension to match dimension of x

    dataset = TensorDataset(Xdat, Ydat) # combining two tensors in one dataset

    kfold = KFold(n_splits=k_folds, shuffle=True) # load kfold function

    for fold, (train, val) in enumerate(kfold.split(dataset)):   # k-fold validation, iterating through dataset and spliting into training and testing sets

      print(f'\nFOLD No. {fold}')
      print("---------------\n\n")

      # Sample elements randomly from a given list of idexes with no replacement.
      train_sample = torch.utils.data.SubsetRandomSampler(train)
      val_sample = torch.utils.data.SubsetRandomSampler(val)

      # data loaders for training and testing
      trainloader = torch.utils.data.DataLoader(dataset, batch_size=20, sampler=train_sample)
      valloader = torch.utils.data.DataLoader(dataset, batch_size=20, sampler=val_sample)

      model = Net()

      # optimizer
      optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

      # training for number of epochs
      for epoch in range(0, num_epochs):

        # Print epoch
        print(f'epoch no. {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

          # Get inputs
          input, target = data

          # reset gradients to zero so it won't accumalate nxt time
          optimizer.zero_grad()

          # Perform forward pass
          output = model(input)

          # Compute RMSE
          loss = torch.sqrt(loss_funct(output, target))

          # Perform backward pass
          loss.backward()

          # Perform optimization
          optimizer.step()

          # calculate loss
          current_loss += loss.item()
          if (i%20 == 19):
            print('RMSE Loss: ', current_loss/ 20)
            print("\n")
            current_loss = 0.0

      # Saving model
      file = 'my_Model.pth'
      torch.save(model.state_dict(), file)

"""**Model Prediction**

"""

# Loading Data
Yts=np.loadtxt("TestData.csv")
Ytsdat = torch.from_numpy(Yts).type(torch.float)

# Loading the model
mod = Net()
predict = mod(Ytsdat)
mod.load_state_dict(torch.load(file))
mod.eval()


p = predict.detach().numpy()
p_df = pd.DataFrame(p)
p_df.to_csv('20I-2003_predictions.csv')

df = pd.read_csv (r'20I-2003_predictions.csv')
print(df)
