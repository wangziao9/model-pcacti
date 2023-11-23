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

def evaluate_current_xml(param_names:List[str]) -> Union[Dict[str, str], str]:
    """
    Runs cacti with the current configuration and parses output.
    For every output parameter in param_names, trys to find the corresponding output, store in a dict and return.
    A "elapsed_time (s)" tracking how long cacti runs under the current config is recorded in the dict in addition.
    If the configuration is invalid, cacti will not return 0 and this function will return cacti's output as string.
    If there is no corresponding output for any output parameter, this function will also return cacti's output as string.
    """
    start_t = time.time()
    completed_process = run_program()
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