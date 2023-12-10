import numpy as np
import copy
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def new_model(model_name, random_seed):
    if model_name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(20, 20), solver='lbfgs', max_iter=5000, random_state=random_seed)
    elif model_name == "KNN":
        return KNeighborsRegressor(n_neighbors=1)
    elif model_name == "SVR":
        return SVR(kernel='rbf', C=100, gamma=1,epsilon=0.1)
    elif model_name =="FOREST":
        return RandomForestRegressor(n_estimators=100, max_depth=30, random_state=random_seed)

def model_specific_input_transform(X_train, X_test, model_name, output_name):
    if model_name == "KNN":
        X_train_dup = copy.deepcopy(X_train)
        X_train_dup[:,0] = X_train_dup[:,0]*0.001
        X_test_dup = copy.deepcopy(X_test)
        X_test_dup[:,0] = X_test_dup[:,0]*0.001
        return X_train_dup, X_test_dup
    elif model_name == "SVR":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        if output_name == "Access time (ns)" or output_name == "Cycle time (ns)":
            X_train_scaled[:,0] = X_train_scaled[:,0]*0.001
            X_test_scaled[:,0] = X_test_scaled[:,0]*0.001
        return X_train_scaled, X_test_scaled
    else:
        return X_train, X_test

def single_run(model_name, output_name, X_train, X_test, y_train, y_test, random_seed):
    X_train_tf, X_test_tf = model_specific_input_transform(X_train, X_test, model_name, output_name)
    regr = new_model(model_name, random_seed)
    regr.fit(X_train_tf, y_train)
    y_pred = regr.predict(X_test_tf)
    mse = mean_squared_error(y_pred, y_test)
    rmse = np.sqrt(mse)
    coeff_of_determination = regr.score(X_test_tf, y_test)
    return rmse, coeff_of_determination