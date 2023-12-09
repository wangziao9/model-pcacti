"""
Analyzes the running time of Cacti and the ML methods
"""

import math
import time
import timeit
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from collect_data import load_dataframes
from train import get_dataframes, transform_frames, split_train_test
import matplotlib.pyplot as plt

# Analyze running time of cacti
try:
    f_l2 = open("L2_data.csv", 'r')
    csv_title = f_l2.readline().strip().split(',')
    f_l2.close()
    f_l3 = open("L3_data.csv", 'r')
    f_l3.close()
except FileNotFoundError:
    print("please prepare L2 and L3 data, make sure L2_data.csv, L3_data.csv exists in the current directory")
    
size_idx = csv_title.index("cache_size")
time_idx = csv_title.index("elapsed_time (s)")
for layer in ["L2", "L3"]:
    f_l = open(f"{layer}_data.csv", 'r')
    frames_l2 = load_dataframes(f_l)
    log_sizes = list(range(int(math.log(4096, 2)), int(math.log(131072, 2))+1)) if layer == "L2" else \
                list(range(int(math.log(65536, 2)), int(math.log(2097152, 2))+1))
    all_times = dict()
    for frame in frames_l2:
        cache_size = int(frame[size_idx])
        log_size = int(math.log(cache_size, 2))
        running_time = float(frame[time_idx])
        if log_size in all_times.keys():
            all_times[log_size].append(running_time)
        else:
            all_times[log_size] = [running_time]
    print(f"For {layer} caches:")
    avg_times = list()
    for log_size in log_sizes:
        running_times = all_times[log_size]
        avg_running_time = sum(running_times) / len(running_times)
        avg_times.append(avg_running_time)
        print(f"cache size:{2**log_size}B (log = {log_size}), average cacti running time: {'%.4f'%avg_running_time}(s)")
    plt.plot([2**size for size in log_sizes], avg_times, 'o', label=layer+" cache")
plt.xlabel("Cache size (B)")
plt.xscale('log') # can add basex=2 here optionally
plt.ylabel("Running time (s)")
plt.legend()
plt.title("Average Running time for P-CACTI")
plt.show()

# Analyze running time for our model    
frames = get_dataframes()
X, Y = transform_frames(frames)
y = Y[:,0]
X_train, X_test, y_train, y_test = split_train_test(X, y)
print("Using ML methods to predict %d datapoints"%X_test.shape[0])

def printtime(regr, name):
    def my_inference():
        regr.predict(X_test)
    elapsed_t = timeit.timeit(my_inference, number=50) / 50
    n_evaled = X_test.shape[0]
    print(name+": total time is {:.3g} seconds,".format(elapsed_t)+"average time is {:.3g} seconds.".format(elapsed_t / n_evaled))
    
regr = MLPRegressor(hidden_layer_sizes=(20, 20), solver='lbfgs', max_iter=5000, random_state=0).fit(X_train, y_train)
printtime(regr, "MLP")
regr = KNeighborsRegressor(n_neighbors=1).fit(X_train, y_train)
printtime(regr, "KNN")
regr = SVR(kernel='rbf', C= 100, gamma=0.001,epsilon=0.001, degree=5).fit(X_train, y_train)
printtime(regr, "SVR")
regr = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=0).fit(X_train, y_train)    
printtime(regr, "Forest")