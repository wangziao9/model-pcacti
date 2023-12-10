"""
Experiment using the random split setting.
Investigates how model error relates to the numbers of available training points
Example usage: python batch_experiment.py -s 0 --num_repeat 3 -test_ratio 0.5 -n_devide 10 0
For help: python batch_experiment.py -h
"""

import numpy as np
from sklearn.model_selection import train_test_split
from batch_common import single_run
from train import get_dataframes, transform_frames, output_names
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_idx",help="used to index into this list: "+str(output_names),type=int)
parser.add_argument("-s","--seed",type=int, default=0, help="set random seed for reproducibility")
parser.add_argument("--num_repeat",type=int, default=3, help="number of repeat experiments")
parser.add_argument("--test_ratio",type=float, default=0.1, help="devote this ratio of data to the test set")
parser.add_argument("--n_devide",type=int, default=5, help="program will try using {1/n, 2/n, ..., all} of the training set")
args = parser.parse_args()
output_name = output_names[args.output_idx]
print("Predicting "+output_name)

frames = get_dataframes()
X, Y = transform_frames(frames)
y = Y[:,args.output_idx]
X_train_, _, _, _ = train_test_split(X,y,test_size=args.test_ratio)
n_fraction = len(X_train_) // args.n_devide
n_train = [len(X_train_)-i*n_fraction for i in range(args.n_devide)]

# experiment
results = np.zeros((1+4,args.n_devide,args.num_repeat))
for seed_add in range(args.num_repeat):
    seed = seed_add + args.seed
    print("experimenting with seed ", seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_ratio, random_state=seed, shuffle=True)
    y_variance = ((y_test - y_test.mean())**2).sum() / len(y_test)
    results[0,:,seed_add] = np.sqrt(y_variance)
    for j,model_name in enumerate(["MLP","KNN","SVR","FOREST"]):
        print(" model "+model_name)
        for div in range(args.n_devide):
            rmse, _ = single_run(model_name,output_name,X_train[div*n_fraction:],X_test,y_train[div*n_fraction:],y_test,seed)
            results[1+j,div,seed_add] = rmse

# aggregate
results_avg = np.average(results,axis=-1)
results_sd = np.std(results, axis=-1) # sd (standard deviation)
results_sem = results_sd / np.sqrt(args.num_repeat) # sem (standard error of the mean) is calculated by the square root of repeat times
for j,model_name in enumerate(["MLP","KNN","SVR","FOREST"]):
    plt.errorbar(n_train, results_avg[1+j], yerr=results_sem[1+j], label="r.m.s.e. for "+model_name)
plt.axhline(y=results_avg[0,0], linestyle='--', label="standard deviation of y true")
plt.plot([], [], ' ', label="Number of testing samples: %d"%len(X_test))
plt.xlabel("Number of training samples")
plt.ylabel("Error")
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.title("Predicting " + output_names[args.output_idx])
plt.savefig(["Access","Cycle","Read","Write","Leak"][args.output_idx]+".png",bbox_inches='tight')