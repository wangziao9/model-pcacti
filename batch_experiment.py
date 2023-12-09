"""
Experiment using the random split setting.
Investigates how model error relates to the numbers of available training points
Example usage: python batch_experiment.py -n 10 -r 0.2 0 KNN
For help: python batch_experiment.py -h
"""

import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from train import get_dataframes, transform_frames, output_names
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_idx",help="used to index into this list: "+str(output_names),type=int)
parser.add_argument("ml_model",type=str, choices=["MLP","KNN","SVR","FOREST","ALL"],help="machine learning method used")
parser.add_argument("-s","--seed",type=int, default=0, help="set random seed for reproducibility")
parser.add_argument("-r","--test_ratio",type=float, default=0.1, help="devote this ratio of data to the test set")
parser.add_argument("-n","--n_devide",type=int, default=5, help="program will try using {1/n, 2/n, ..., all} of the training set")
args = parser.parse_args()
print("Predicting "+output_names[args.output_idx])

def new_model(model_name):
    if model_name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(20, 20), solver='lbfgs', max_iter=5000, random_state=0)
    elif model_name == "KNN":
        return KNeighborsRegressor(n_neighbors=1)
    elif model_name == "SVR":
        return SVR(kernel='rbf', C=100, gamma=0.1,epsilon=0.01)
    elif model_name =="FOREST":
        return RandomForestRegressor(n_estimators=100, max_depth=30, random_state=0)

def model_specific_input_transform(X, model_name):
    if model_name == "KNN":
        X[:,0] = X[:,0]*0.001

frames = get_dataframes()
X, Y = transform_frames(frames)
y = Y[:,args.output_idx]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_ratio, random_state=args.seed, shuffle=True)
y_variance = ((y_test - y_test.mean())**2).sum() / len(y_test)
print("The testing set contains " + str(len(X_test)) + " samples, has variance %.4f"%y_variance, ", std %.4f"%np.sqrt(y_variance))
print("The training set contains " + str(len(X_train)) + " samples")
n_fraction = len(X_train) // args.n_devide
n_train = [len(X_train)-i*n_fraction for i in range(args.n_devide)]

def experiment_with_model(model_name):
    model_specific_input_transform(X_train, model_name)
    model_specific_input_transform(X_test, model_name)
    rmse_store = list()
    for i in range(args.n_devide):
        print("Training a "+model_name+" model with "+str(len(X_train)-i*n_fraction)+" samples")
        regr = new_model(model_name)
        regr.fit(X_train[i*n_fraction:], y_train[i*n_fraction:])
        y_pred = regr.predict(X_test)
        mse = mean_squared_error(y_pred, y_test)
        rmse = np.sqrt(mse)
        coeff_of_determination = regr.score(X_test, y_test)
        print("coef. of determination: %.4f"%coeff_of_determination + ", rmse:%.4f"%rmse)
        rmse_store.append(rmse)
    return rmse_store

for model_name in ["MLP","KNN","SVR","FOREST"] if args.ml_model=="ALL" else [args.ml_model]:
    rmse_store = experiment_with_model(model_name)
    plt.plot(n_train, rmse_store, '-x', label="r.m.s. error for "+model_name)
plt.axhline(y=np.sqrt(y_variance), linestyle='--', label="standard deviation of y true")
plt.plot([], [], ' ', label="Number of testing samples: %d"%len(X_test))
plt.xlabel("Number of training samples")
plt.ylabel("Error")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.title("Using "+args.ml_model+" to predict "+output_names[args.output_idx])
plt.show()