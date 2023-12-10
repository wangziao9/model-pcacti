# Basically copy-and-pasted from https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# Edited by us

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class CactiDataset(torch.utils.data.Dataset):
    '''
    Prepare the Boston dataset for regression
    '''

    def __init__(self, X, y, scale_data=True):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            # Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class CactiNet(nn.Module):

    def __init__(self):
        super(CactiNet, self).__init__()

        self.layers = nn.Sequential(
            # 11 inputs (including 1-hot encoding for Cache Level and Access Mode),
            # Let's say it learns... 33 outputs?
            # nn.Linear(11, 33),
            # nn.ReLU(),
            
            # # 33 in from last layer, output 66
            # nn.Linear(33, 66),
            # nn.ReLU(),

            # # Add a few stronger layers?
            # nn.Linear(66, 100),
            # nn.ReLU(),

            # nn.Linear(100, 100),
            # nn.ReLU(),

            # nn.Linear(100, 66),
            # nn.ReLU(),

            # # 66 in from last layer, output 33 again
            # nn.Linear(66, 33),
            # nn.ReLU(),

            # # 33 in from last layer, output 5 values...
            # nn.Linear(33, 5)
            nn.Linear(11, 22),
            nn.ReLU(),
            
            # 33 in from last layer, output 66
            nn.Linear(22, 33),
            nn.ReLU(),

            # Add a few stronger layers?
            nn.Linear(33, 44),
            nn.ReLU(),

            nn.Linear(44, 55),
            nn.ReLU(),

            nn.Linear(55, 55),
            nn.ReLU(),

            nn.Linear(55, 44),
            nn.ReLU(),

            nn.Linear(44, 33),
            nn.ReLU(),

            # 33 to 22
            nn.Linear(33, 22),
            nn.ReLU(),

            # 22 to 11
            nn.Linear(22, 11),
            nn.ReLU(),

            # 11 to 5, finish
            nn.Linear(11, 5)
        )

    def forward(self, x):
        '''
        Forward pass:
        '''
        return self.layers(x)


# net = CactiNet()
# print(net)

# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # The weight for first fully conected layer.