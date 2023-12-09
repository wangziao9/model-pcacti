import subprocess
import xml.etree.ElementTree as ET
import os
import time
from typing import List, Dict, Union

tree = ET.parse("ref.xml")
root = tree.getroot()

def run_program() -> subprocess.CompletedProcess:
    tree.write("tmp.xml")
    os.chdir("../pcacti/")
    result = subprocess.run(["./cacti", "-infile", "../model-pcacti/tmp.xml"], \
                          check=False, capture_output=True, encoding="utf8")
    os.chdir("../model-pcacti")
    # os.remove("tmp.xml")
    return result

def run_program_cmdarg() -> subprocess.CompletedProcess:
    default_tag = (xml_get("tag_size") == "default")
    access_mode_d = {"normal":"0", "sequential":"1", "fast":"2"}
    is_cache = (xml_get("memory_type")=="cache")
    is_main_mem = (xml_get("memory_type")=="main memory")
    ed_ed2_none_d = {"ED":"0", "ED^2":"1", "None":"2"}
    projection_d = {"aggressive":"0", "conservative":"1"}
    cache_model_d = {"UCA":"0", "NUCA":"1"}
    cache_level_d = {"L2":"0", "L3":"1"}
    myargs = ["./cacti",\
        xml_get("cache_size"),\
        xml_get("block_size"),\
        xml_get("associativity"),\
        xml_get("ports.read_write_port"),\
        xml_get("ports.exclusive_read_port"),\
        xml_get("ports.exclusive_write_port"),\
        xml_get("ports.single_ended_read_ports"),\
        xml_get("uca_bank_count"),\
        str(1000*float(xml_get("technology_node"))),\
        xml_get("page_size"),\
        xml_get("burst_length"),\
        xml_get("internal_prefetch_width"),
        xml_get("bus_width"),\
        "0" if default_tag else "1",\
        "42" if default_tag else xml_get("tag_size"),\
        access_mode_d[xml_get("access_mode")],\
        "1" if is_cache else "0",\
        "1" if is_main_mem else "0",\
        xml_get("objective_function.design_objective.weights.delay"),\
        xml_get("objective_function.design_objective.weights.dynamic_power"),\
        xml_get("objective_function.design_objective.weights.leakage_power"),\
        xml_get("objective_function.design_objective.weights.area"),\
        xml_get("objective_function.design_objective.weights.cycle_time"),\
        xml_get("objective_function.design_objective.deviations.delay"),\
        xml_get("objective_function.design_objective.deviations.dynamic_power"),\
        xml_get("objective_function.design_objective.deviations.leakage_power"),\
        xml_get("objective_function.design_objective.deviations.area"),\
        xml_get("objective_function.design_objective.deviations.cycle_time"),\
        ed_ed2_none_d[xml_get("objective_function.optimize")],\
        xml_get("temperature"),\
        "3",\
        "0","0","0","0",\
        projection_d[xml_get("interconnects.projection")],\
        "2","2",\
        cache_model_d[xml_get("cache_model")],\
        xml_get("core_count"),\
        cache_level_d[xml_get("cache_level")],\
        xml_get("nuca_bank_count"),\
        xml_get("objective_function.nuca_design_objective.weights.delay"),\
        xml_get("objective_function.nuca_design_objective.weights.dynamic_power"),\
        xml_get("objective_function.nuca_design_objective.weights.leakage_power"),\
        xml_get("objective_function.nuca_design_objective.weights.area"),\
        xml_get("objective_function.nuca_design_objective.weights.cycle_time"),\
        xml_get("objective_function.nuca_design_objective.deviations.delay"),\
        xml_get("objective_function.nuca_design_objective.deviations.dynamic_power"),\
        xml_get("objective_function.nuca_design_objective.deviations.leakage_power"),\
        xml_get("objective_function.nuca_design_objective.deviations.area"),\
        xml_get("objective_function.nuca_design_objective.deviations.cycle_time"),\
        "1","false"
    ]
    print("length of myargs:",len(myargs))
    os.chdir("../pcacti/")
    result = subprocess.run(myargs, check=False, capture_output=True, encoding="utf8")
    os.chdir("../model-pcacti")
    return result

def xml_set(xml_name:str, xml_value) -> bool:
    "To modify value inside <a><b>value</b></a>, pass \"a.b\" as xml_name"
    elem = root
    for name in xml_name.split("."):
        elem = elem.find(name)
    if elem is None:
        return False
    else:
        elem.text = str(xml_value)
        return True

def xml_get(xml_name:str) -> Union[str, None]:
    elem = root
    for name in xml_name.split("."):
        elem = elem.find(name)
    if elem is None:
        return False
    else:
        return elem.text

def evaluate_current_xml(param_names:List[str], use_cmd_args=False) -> Union[Dict[str, str], str]:
    """
    Runs cacti with the current configuration and parses output.
    For every output parameter in param_names, trys to find the corresponding output, store in a dict and return.
    A "elapsed_time (s)" tracking how long cacti runs under the current config is recorded in the dict in addition.
    If the configuration is invalid, cacti will not return 0 and this function will return cacti's output as string.
    If there is no corresponding output for any output parameter, this function will also return cacti's output as string.
    """
    start_t = time.time()
    completed_process = run_program_cmdarg() if use_cmd_args else run_program()
    end_t = time.time()
    if completed_process.returncode != 0:
        return " \\n ".join(completed_process.stdout[:50].split('\n'))
    ret = dict()
    for line in completed_process.stdout.split('\n'):
        line = line.split(":")
        if len(line) != 2:
            continue
        key = line[0].strip()
        if key in param_names:
            value = line[1].strip()
            ret[key] = value
    if len(ret) == 0:
        return " \\n ".join(completed_process.stdout[:50].split('\n'))
    ret["elapsed_time (s)"] = "{:.3g}".format(end_t - start_t)
    return ret