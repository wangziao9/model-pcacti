"""
After running the batch_technode to generate the csvs files,
run this script to generate all the figures for predicting unknown tech node experiment
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
results = np.zeros((5,1+4,7))
output_names = [None]*5
target_technodes = [0.014, 0.016, 0.022, 0.032, 0.045, 0.065, 0.090]
for itech,test_node in enumerate(target_technodes):
    f_in = open("technode_experiment_"+str(test_node)+".csv",'r')
    reader = csv.reader(f_in)
    reader.__next__()
    for output_idx,row in enumerate(reader):
        for ient,value in enumerate(row):
            if ient == 0:
                output_names[output_idx] = value
                continue
            results[output_idx,ient-1,itech] = float(value)
    f_in.close()

for output_idx, output_name in enumerate(output_names):
    for j,model_name in enumerate(["MLP","KNN","SVR","FOREST"]):
        plt.plot(target_technodes, results[output_idx,1+j,:],'-x',label="r.m.s.e. for "+model_name)
    # plt.plot(target_technodes,results[output_idx,0,:],'-x',label="standard deviation of y true")
    plt.xlabel("Target Technode (nm)")
    plt.ylabel("Error")
    plt.xticks(target_technodes, [str(int(1000*node)) for node in target_technodes],fontsize=8)
    plt.xlim(left=0.012)
    plt.ylim(bottom=0)
    plt.legend()
    plt.title("Predicting " + output_name)
    plt.savefig("Technode_"+["Access","Cycle","Read","Write","Leak"][output_idx]+".png",bbox_inches='tight')
    plt.close()