"""
Selects a target tech node, train model on all available data that is not from that tech node, and tests on the target tech node.
Performs the ablve experiment for every output catagory (individually) and every model.
Outputs a csv file, row is output catagory, column is model used.
Repeats every experiment and averages result.
Example usage: python batch_technode.py -s 0 -n 3 -t 0.032
For help: python batch_technode.py -h
"""

import csv
import numpy as np
from sklearn.model_selection import train_test_split
from batch_common import single_run
from train import get_dataframes, transform_frames, output_names

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-s","--seed",type=int, default=0, help="set random seed for reproducibility")
parser.add_argument("-n","--num_repeat",type=int, default=3, help="number of repeat experiments")
parser.add_argument("-t","--test_node",type=float, default=0.032, help="tech node for test set")
args = parser.parse_args()

# load and split data
frames = get_dataframes()
X, Y = transform_frames(frames)

# run experiment
results = np.zeros((5,1+4,args.num_repeat))
for seed_add in range(args.num_repeat):
    print("experimenting with seed ", args.seed+seed_add)
    np.random.seed(args.seed+seed_add)
    permute = np.random.permutation(len(X))
    X_, Y_ = X[permute], Y[permute]
    test_indices = (np.abs(X_[:,0] - args.test_node*1000) < 1e-5)
    X_train, Y_train = X_[~test_indices], Y_[~test_indices]
    X_test, Y_test = X_[test_indices], Y_[test_indices]
    for output_idx in range(5):
        print("output idx = ", output_idx)
        y_train, y_test = Y_train[:,output_idx], Y_test[:,output_idx]
        y_variance = ((y_test - y_test.mean())**2).sum() / len(y_test)
        results[output_idx,0,seed_add]=y_variance
        for j,model_name in enumerate(["MLP","KNN","SVR","FOREST"]):
            rmse, _ = single_run(model_name, output_names[output_idx], X_train, X_test, y_train, y_test, args.seed+seed_add)
            results[output_idx,1+j,seed_add]=rmse

# write to csv
results_avg = np.average(results,axis=-1)
f_out = open("technode_experiment_"+str(args.test_node)+".csv",'w',newline='')
writer = csv.writer(f_out)
writer.writerow(["Output", "s.d. of true"]+["MLP","KNN","SVR","FOREST"])
for output_idx in range(5):
    writer.writerow([output_names[output_idx]] + ["%.4f"%err for err in results_avg[output_idx]])
f_out.close()