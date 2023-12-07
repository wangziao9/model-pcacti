import math
import numpy as np
from collect_data import inputs, input_names, output_names, get_dataframes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from typing import Union, List, Tuple

# Needed for larger outputs
import csv

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
out_select = parse_next_config(settings_file, "Output Select")
config_output_idx = [output_names.index(i) for i in out_select]
# print(f"DEBUG: config_output_idx = {config_output_idx}")
config_method = parse_next_config(settings_file, "Method")[0]
config_multi_r = parse_next_config(settings_file, "Multiple Results")[0]
config_savefile = parse_next_config(settings_file, "Savefile Name")[0]

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
    if y.shape[1] == 1:
        y = np.ravel(y)
    print("X.shape: ", X.shape, " y.shape: ", y.shape)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Check if we're collecting multiple outputs
    if config_multi_r == "True":
        # create a list of inputs to use
        # [hidden_layer_sizes, solver], [n_neighbors, weights]
        # First is if MLPRegressor, Second is if KNN
        # Check which model we're using
        if config_method == "MLP":
            hyper_params = [[(20, 20), 'lbfgs'],
                            [(50, 50), 'lbfgs'],
                            [(100, 100), 'lbfgs'],
                            [(20, 20, 20), 'lbfgs'],
                            [(50, 50, 50), 'lbfgs'],
                            [(100, 100, 100), 'lbfgs'],
                            [(20, 20, 20, 20), 'lbfgs'],
                            [(50, 50, 50, 50), 'lbfgs'],
                            [(100, 100, 100, 100), 'lbfgs'],
                            [(20, 20), 'adam'],
                            [(50, 50), 'adam'],
                            [(100, 100), 'adam'],
                            [(20, 20, 20), 'adam'],
                            [(50, 50, 50), 'adam'],
                            [(100, 100, 100), 'adam'],
                            [(20, 20, 20, 20), 'adam'],
                            [(50, 50, 50, 50), 'adam'],
                            [(100, 100, 100, 100), 'adam']
                            ]
            # output_data = [["Model Type", "Number of Hidden Layers", "Hidden Layer Width", "Solver", "P. Access Time", "P. Cycle Time", "P. Dyn. Read", "P. Dyn. Write", "P. Leakage", "MSE", "Root MSE", "Variance GT", "STD GT", "Coef. of Determ."]]
            output_data = [["Model Type", "Number of Hidden Layers", "Hidden Layer Width", "Solver", "Outputs Predicted", "MSE", "Root MSE", "Variance GT", "STD GT", "Coef. of Determ."]]
        else:
            hyper_params = [[1, 'uniform'],
                            [2, 'uniform'],
                            [2, 'distance'],
                            [3, 'uniform'],
                            [3, 'distance'],
                            [4, 'uniform'],
                            [4, 'distance'],
                            [5, 'uniform'],
                            [5, 'distance']
                            ]
            # output_data = [["Model Type", "Number of Neighbors", "Weights", "P. Access Time", "P. Cycle Time", "P. Dyn. Read", "P. Dyn. Write", "P. Leakage", "MSE", "Root MSE", "Variance GT", "STD GT", "Coef. of Determ."]]
            output_data = [["Model Type", "Number of Neighbors", "Weights", "Outputs Predicted", "MSE", "Root MSE", "Variance GT", "STD GT", "Coef. of Determ."]]
            print(f"DEBUG - number of neighbors(before training): {[hyper_params[i][0] for i in range(len(hyper_params))]}")
    else:
        # Just the defaults
        if config_method == "MLP":
            hyper_params = [[(20, 20), 'lbfgs']]
        else:
            hyper_params = [1, 'uniform']

    for ind in range(len(hyper_params)):
        # train model and predict
        print("Training a "+ config_method)
        if config_method == "MLP":
            # Consider tuning the hidden_layer_sizes, solver and max_iter.
            hl_sizes = hyper_params[ind][0]
            curr_solver = hyper_params[ind][1]
            regr = MLPRegressor(hidden_layer_sizes=hl_sizes, solver=curr_solver, max_iter=1000, random_state=config_random_seed).fit(X_train, y_train)
        elif config_method == "KNN":
            # Get out hyper parameters
            num_neigh = hyper_params[ind][0]
            print(f"DEBUG - Number of Neighbors: {num_neigh}")
            k_weights = hyper_params[ind][1]
            print(f"DEBUG - Weights: {k_weights}")
            regr = KNeighborsRegressor(n_neighbors=num_neigh, weights=k_weights).fit(X_train, y_train)
        else:
            print("Method not supported, exiting"); exit(0)
        y_pred = regr.predict(X_test)
        # evaluate results
        print("Predicting "+str(out_select))
        mse = mean_squared_error(y_pred, y_test) # We let it return the average of the MSEs for each prediction target
        # print(f"DEBUG: MSE without uniform averaging: {mean_squared_error(y_pred, y_test, multioutput='raw_values')}")
        rmse = np.sqrt(mse)
        print("MSE (Mean Squared Error): {:.4g}, Root MSE: {:.4g}".format(mse,rmse))
        y_var_list = np.array([])
        if len(y_test.shape) > 1:
            # print(f"---\n\n---\nDEBUG: y_test[:,0] = {y_test[:,0]}")
            for i in range(y_test.shape[1]):
                y_var_list = np.append(y_var_list, ((y_test[:,i] - y_test[:,i].mean())**2).sum() / y_test.shape[0])
        else:
            y_var_list = np.array(((y_test - y_test.mean())**2).sum() / len(y_test))
        # print(f" DEBUG: y_var_list: {y_var_list}")
        y_variance = y_var_list.mean()
        # print(f"  DEBUG: y_test.shape = {y_test.shape}")
        # print(f"  DEBUG: (y_test - y_test.mean()) = {y_test - y_test.mean()}")
        # print(f"  DEBUG: (y_test - y_test.mean()**2).sum() = {((y_test - y_test.mean())**2).sum()}")
        # print(f"  DEBUG: len(y_test) = {len(y_test)}")
        y_std = np.sqrt(y_variance)
        print("Variance and Standard Deviation of Ground Truth: {:.4g}, {:.4g}".format(y_variance, y_std))
        coeff_of_determination = regr.score(X_test, y_test)
        print(f"DEBUG - coefficient_of_determination:\n {coeff_of_determination}")
        print(f"DEBUG - (1-mse/y_variance):\n {1-mse/y_variance}")
        #assert(np.abs(coeff_of_determination - (1-mse/y_variance)) < 1e-4)
        assert(np.abs(coeff_of_determination - (1-mse/y_variance)) < 5e-3)
        print("Coefficient of determination: {:.4g}".format(coeff_of_determination))

        if config_multi_r == "True":
            if len(out_select) == 5:
                shown_out_select = 'all'
            else:
                shown_out_select = out_select
            if config_method == "MLP":
                output_data.append(["MLP", len(hyper_params[ind][0]), hyper_params[ind][0][0], hyper_params[ind][1], shown_out_select, mse, rmse, y_variance, y_std, coeff_of_determination])
            elif config_method == "KNN":
                print(f"DEBUG - hyper_params[{ind}][0] = {hyper_params[ind][0]}")
                output_data.append(["KNN", hyper_params[ind][0], hyper_params[ind][1], shown_out_select, mse, rmse, y_variance, y_std, coeff_of_determination])

    if config_multi_r == "True":
        with open(str(config_savefile) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(output_data[0])
            writer.writerows(output_data[1:])
