import math
import numpy as np
from collect_data import inputs, input_names, output_names, get_dataframes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Union, List, Tuple
import torch
from torch import nn
from torchmod import CactiDataset
from torchmod import CactiNet


settings_file = open("settings.cfg",'r')
while settings_file.__next__()!="Setup\n":
    pass

def parse_next_config(f,name):
    split_line = f.__next__().split(":")
    assert split_line[0].strip() == name
    split_line = split_line[1].split("#")[0]
    split_line = split_line.split(",")
    return [cfgval.strip() for cfgval in split_line]

config_random_seed = int(parse_next_config(settings_file, "Random State")[0])
config_split_method, config_split_argument2 = parse_next_config(settings_file, "Train Test Split Method")
#config_output_idx = output_names.index(parse_next_config(settings_file, "Output Select")[0])
out_select = parse_next_config(settings_file, "Output Select")
config_output_idx = [output_names.index(i) for i in out_select]
config_method = parse_next_config(settings_file, "Method")[0]
config_paramsearch = parse_next_config(settings_file, "Param Search")[0]
bayes_search_niter = 5

def input_transforms(name:str, value:str) -> Union[float, List[float]]:
    if name == "technology_node":
        return 1000 * float(value)
    if name == "cache_size" or name == "associativity":
        return math.log(int(value), 2)
    if name == "ports.exclusive_read_port" or name == "ports.exclusive_write_port":
        return float(value)
    if name == "uca_bank_count":
        return math.log(int(value), 2)
    if name == "access_mode":
        d = {"normal":[1,0,0], "sequential":[0,1,0], "fast":[0,0,1]}
        return d[value]
    if name == "cache_level": # take into account if L2 or L3
        d = {"L2":[1,0], "L3":[0,1]}
        return d[value]

def transform_frames(frames: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    X, Y = list(), list()
    for frame in frames:
        X_row, Y_row = list(), list()
        for i,name in enumerate(input_names):
            transformed = input_transforms(name, frame[i])
            if isinstance(transformed, List):
                X_row.extend(transformed)
            else:
                X_row.append(transformed)
        for i,_ in enumerate(output_names):
            Y_row.append(float(frame[i+len(input_names)]))
        X.append(X_row); Y.append(Y_row)
    return np.array(X), np.array(Y)

def split_train_test(X, Y):
    "Splits data according to settings.cfg and shuffles"
    if config_split_method == "Random Split":
        test_ratio = float(config_split_argument2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=config_random_seed)
    else:
        # manually shuffle data
        np.random.seed(config_random_seed)
        permute = np.random.permutation(len(X))
        X, Y = X[permute], Y[permute]
        target_node = input_transforms("technology_node", config_split_argument2)
        test_indices = (np.abs(X[:,0] - target_node) < 1e-6)
        X_train, Y_train = X[~test_indices], Y[~test_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]
    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__":
    # preprocess data
    frames = get_dataframes()
    X, Y = transform_frames(frames)
    y = Y[:,config_output_idx]
    print("X[0]: ", X[0], " y[0]: ", y[0])
    print("X.shape: ", X.shape, " y.shape: ", y.shape)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    # train model and predict

    dataset = CactiDataset(X_train, y_train)
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=30, shuffle=True, num_workers=1)
  
    # Initialize the MLP
    cnet = CactiNet()
    
    # Define the loss function and optimizer
    #loss_function = nn.L1Loss()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(cnet.parameters(), lr=1e-4)
  
    # Run the training loop
    for epoch in range(0, 15): # 5 epochs at maximum
        
        # Print epoch
        print(f'Starting epoch {epoch+1}')
        
        # Set current loss value
        current_loss = 0.0
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float(), targets.float()
            # if i % 10 == 0:
            #     print(f"DEBUG - Inputs is {inputs}")
            # print(f"DEBUG - Targets is {targets}")
            targets = targets.reshape((targets.shape[0], 5))
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = cnet(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 0:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0

    # Training is complete.
    print('Training process has finished.')

    pred_dataset = CactiDataset(X_test, y_test)
    
    predloader = torch.utils.data.DataLoader(pred_dataset, batch_size=10, shuffle=True, num_workers=1)

    cnet.eval()

    # Get the two tensors ready
    y_gt = torch.zeros(1,5)
    y_predicted = torch.zeros(1,5)

    # Iterate over the DataLoader for training data
    for i, data in enumerate(predloader, 0):
        
        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        #print(f"DEBUG - Inputs is {inputs}")
        #print(f"DEBUG - Targets is {targets}")
        targets = targets.reshape((targets.shape[0], 5))
        #print(f"DEBUG - After reshape Targets is {targets}")

        y_gt = torch.cat((y_gt, targets), 0)
        
        # Make prediction
        outputs = cnet(inputs)
        #print(f"DEBUG - Outputs: {outputs}")
        y_predicted = torch.cat((y_predicted, outputs), 0)

        # Compute loss
        loss = loss_function(outputs, targets)

        # # Compute MSE for this bit?
        # temp_y_gt = targets.detach().numpy()
        # temp_y_pred = outputs.detach().numpy()
        # temp_mse = mean_squared_error(temp_y_gt, temp_y_pred)

        # if temp_mse > 30.0:
        #     print(f"For mini-batch {i}, got MSE {temp_mse:.3f}!")
        #     print(f"Outputs:\n {outputs}")
        #     print(f"Ground Truth:\n {targets}")
        
        # Print statistics
        current_loss += loss.item()
        if i % 10 == 0:
            print('Loss after mini-batch %5d: %.3f' %
                (i + 1, current_loss / 500))
            current_loss = 0.0
        
    num_y_gt = y_gt.detach().numpy()[1:]
    num_y_predicted = y_predicted.detach().numpy()[1:]

    print(f"First 3 entries of Ground Truth:\n {num_y_gt[:3]}\n")
    print(f"First 3 entries of Predicted:\n {num_y_predicted[:3]}\n")

    mse = mean_squared_error(num_y_gt, num_y_predicted)

    # We're done.
    print(f"Mean Squared Error on test set is: {mse:.3f}")
