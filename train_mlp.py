import math
import numpy as np
from collect_data import inputs, input_names, output_names, get_dataframes
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from typing import Union, List, Tuple

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
config_output_idx = output_names.index(parse_next_config(settings_file, "Output Select")[0])

def input_transforms(name:str, value:str) -> Union[float, List[float]]:
    if name == "technology_node":
        return float(value)
    if name == "cache_size" or name == "associativity":
        return math.log(int(value), 2)
    if name == "ports.exclusive_read_port" or name == "ports.exclusive_write_port":
        return float(value)
    if name == "uca_bank_count":
        return math.log(int(value), 2)
    if name == "access_mode":
        d = {"normal":[1,0,0], "sequential":[0,1,0], "fast":[0,0,1]}
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
    print(X[0], y[0])
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    # train model and predict
    print(y_test)
    regr = MLPRegressor(random_state=config_random_seed, max_iter=500).fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    # evaluate results
    print("Predicting "+output_names[config_output_idx])
    mse = mean_squared_error(y_pred, y_test)
    rmse = np.sqrt(mse)
    print("MSE (Mean Squared Error): {:.4g}, Root MSE: {:.4g}".format(mse,rmse))
    y_variance = ((y_test - y_test.mean())**2).sum() / len(y_test)
    y_std = np.sqrt(y_variance)
    print("Variance and Standard Deviation of Ground Truth: {:.4g}, {:.4g}".format(y_variance, y_std))
    coeff_of_determination = regr.score(X_test, y_test)
    assert(np.abs(coeff_of_determination - (1-mse/y_variance)) < 1e-4)
    print("Coefficient of determination: {:.4g}".format(coeff_of_determination))