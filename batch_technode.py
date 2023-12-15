"""
Selects a target tech node, train model on all available data that is not from that tech node, and tests on the target tech node.
Performs the above experiment individually for each tech node and every model.
Repeats every experiment and averages result.
Example usage: python batch_technode.py -s 0 -n 3 0
For help: python batch_technode.py -h
"""

import numpy as np
from batch_common import single_run
from train import get_dataframes, transform_frames, output_names
import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_idx",help="used to index into this list: "+str(output_names),type=int)
parser.add_argument("-s","--seed",type=int, default=0, help="set random seed for reproducibility")
parser.add_argument("-n","--num_repeat",type=int, default=3, help="number of repeat experiments")
args = parser.parse_args()

# load and split data
frames = get_dataframes()
X, Y = transform_frames(frames)

# run experiment
target_technodes = [0.014, 0.016, 0.022, 0.032, 0.045, 0.065, 0.090]
results = np.zeros((1+4,len(target_technodes),args.num_repeat))
for seed_add in range(args.num_repeat):
    print("experimenting with seed ", args.seed+seed_add)
    np.random.seed(args.seed+seed_add)
    permute = np.random.permutation(len(X))
    X_, Y_ = X[permute], Y[permute]
    for tech_idx, test_node in enumerate(target_technodes):
        print("tech node = %.4f"%test_node)
        test_indices = (np.abs(X_[:,0] - test_node*1000) < 1e-5)
        X_train, Y_train = X_[~test_indices], Y_[~test_indices]
        X_test, Y_test = X_[test_indices], Y_[test_indices]
        y_train, y_test = Y_train[:,args.output_idx], Y_test[:,args.output_idx]
        y_variance = ((y_test - y_test.mean())**2).sum() / len(y_test)
        results[0,tech_idx,seed_add]=y_variance
        for j,model_name in enumerate(["MLP","KNN","SVR","FOREST"]):
            rmse, _ = single_run(model_name, output_names[args.output_idx], X_train, X_test, y_train, y_test, args.seed+seed_add)
            results[1+j,tech_idx,seed_add]=rmse

# aggregate
results_avg = np.average(results,axis=-1)
results_sd = np.std(results, axis=-1) # sd (standard deviation)
results_sem = results_sd / np.sqrt(args.num_repeat-1) # sem (standard error of the mean) is calculated by the square root of repeat times
for j,model_name in enumerate(["MLP","KNN","SVR","FOREST"]):
    plt.errorbar(target_technodes, results_avg[1+j], yerr=results_sem[1+j], label="r.m.s.e. for "+model_name)
plt.plot([], [], ' ', label="Repeated %d times"%args.num_repeat)
plt.xlabel("Target Technode (nm)")
plt.ylabel("Error")
plt.xticks(target_technodes, [str(int(1000*node)) for node in target_technodes],fontsize=8)
plt.xlim(left=0.012)
plt.ylim(bottom=0)
plt.legend()
plt.title("Predicting " + output_names[args.output_idx])
plt.savefig("Technode_"+["Access","Cycle","Read","Write","Leak"][args.output_idx]+".png",bbox_inches='tight')