import csv
import os
from run_cacti import xml_set, xml_get, evaluate_current_xml
from typing import List, Tuple

settings_file = open("settings.cfg",'r')
inputs = list()
input_names = list()
output_names = list()
assert(settings_file.readline() == "Input\n")
for line in settings_file:
    if line == "\n":
        break
    line = line.split('#')[0]
    line = line.split(',')
    inputs.append([item.strip() for item in line])
    input_names.append(line[0].strip())
assert(settings_file.readline() == "Output\n")
for line in settings_file:
    if line == "\n":
        break
    line = line.split('#')[0]
    output_names.append(line.strip())

def load_dataframes(f):
    ret = list()
    reader = csv.reader(f)
    reader.__next__()
    for row in reader:
        ret.append(row)
    return ret

def save_dataframes(f, data, fieldnames):
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    writer.writerows(data)

def collect_data(ith:int) -> Tuple[List[str], List[str]]:
    """
    Collects data from cacti, exploring every combination of the input space as specified by settings.cfg.
    In case of error, the input parameters and cacti error message is also recorded.
    This is a recursive function, ith is the recursion depth.
    """
    param = inputs[ith]
    xml_name = param[0]
    ret = list()
    errs = list()
    print("Collecting data from PCACTI...")
    for xml_val in param[1:]:
        if ith == 0:
            print(xml_name, xml_val) # print progress
        if ith == 1:
            print(xml_name, "\t"+xml_val)
        xml_set(xml_name, xml_val)
        if ith == len(inputs) - 1:
            row = list()
            for name in input_names:
                row.append(xml_get(name))
            eval_results = evaluate_current_xml(output_names)
            if isinstance(eval_results, str):
                errs.append(row+[eval_results])
                continue
            try:
                for name in output_names:
                    row.append(eval_results[name])
            except KeyError as e:
                errs.append(row+[str(e)])
                continue
            row.append(eval_results["elapsed_time (s)"])
            ret.append(row)
        else:
            ret_, errs_ = collect_data(ith+1)
            ret.extend(ret_); errs.extend(errs_)
    return ret, errs

def get_dataframes() -> List[List[str]]:
    """
    Gets rows of input output data where each row corresponds to one configuration.
    If "data.csv" does not exist under the current directory, collects data from cacti, and generates data.csv (for valid configs) and errs.csv (for invalid configs).
    If "data.csv" exists under the current directory, uses its data. You must delete "data.csv" if it is outdated.
    """
    if os.path.exists("data.csv"):
        f = open("data.csv",'r')
        frames = load_dataframes(f)
    else:
        #collect_data
        frames, errs = collect_data(0)
        f = open("data.csv",'w')
        save_dataframes(f, frames, input_names + output_names + ["elapsed_time (s)"])
        f.close()
        if len(errs) > 0:
            fe = open("errs.csv",'w')
            save_dataframes(fe, errs, input_names + ["error_type"])
            fe.close()
    return frames