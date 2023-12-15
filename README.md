# model-pcacti
Replace the cache modeling tool "PCACTI" with machine learning models.

## Environment
PCACTI can be downloaded from this source (https://sportlab.usc.edu/downloads/download/), select PACKAGE - PCACTI on the webpage. You need to extract PCACTI into a directory named "pcacti" and run `make` under "pcacti" to build the tool. 

Put this repository aside to "pcacti", see below for directory structure.

```
..
|
+-- pcacti
|     |
|     +--README
|     +--makefile
|     +--...
|
|-- model-pcacti
|     |
|     +--README.md
|     +--...
```

A Python environment with packages like scikit-learn and matplotlib is required for this project.

## How to run our code
First, specify your settings in settings.cfg. For each line under "Input", specify the name of the cache configs to vary (as they appear in ref.xml) and their permissible values. For each line under "Output", specify the name of a output attribute you want to collect (as they appear in PCACTI's stdout). For each line under "Setup", specify the how the machine learning experiment is run, see comments in settings.cfg file.

Run `python train.py` to collect data, train model and evaluate model. The first time you run the program, data will be collected from PCACTI and written to data.csv, all combinations of cache config is explored. This might take a long time. An errs.csv is also generated containing invalid cache configurations that we attempted to use. To trigger (re-)collection of data from PCACTI, you must delete "data.csv" or make sure it doesn't exist (or else the cached data in data.csv is used).

You can concatenate different data.csv files together to aggregate data, as long as they are generated using two settings.cfg files whose only difference is the permissible values of "Input". We used this method to aggregate data for L2 and L3 level cache, whose range of permissible cache sizes are different, into one data.csv.

Currently, we implemented four models: Multi-layer Perceptron regressor and K-nearest neighbors, Support Vector Regressor and Random Forest. train.py also supports hyperparameter search, for that, set "Param Search" in the "Setup" section in settings.cfg to True before running `python train.py`. analyze_running_time.py analyzes and compares running time of cacti and our models. batch_experiment.py runs a series of experiments in the random train-test-split setting. batch_technode.py runs a series of experiments in the split according to tech node setting. See comments at the top of the each source file for how to run them in the command line.

## Optional modifications to PCACTI

PCACTI supports passing cache configuration through a xml file or command line arguments. This project passes the configurations through a xml file by default. If you want to try passing cache configurations through command line arguments, please add the following lines to PCACTI's source code, in file io.cc, function cacti_interface just before line 848 (`init_tech_params(g_ip->F_sz_um, false);`), and recompile:

```
strcpy(g_ip->data_array_cell_tech_file,"xmls/devices/cmos_14nm_std.xml");
strcpy(g_ip->data_array_peri_tech_file,"xmls/devices/cmos_14nm_std.xml");
strcpy(g_ip->tag_array_cell_tech_file,"xmls/devices/cmos_14nm_std.xml");
strcpy(g_ip->tag_array_peri_tech_file,"xmls/devices/cmos_14nm_std.xml");
```

Note that this is not required for the rest of the project to run. You can use the function "run_program_cmdarg" in file run_cacti.py to call cacti passing cache configurations in the command line arguments - this is not the default behavior.

To profile PCACTI, run `make DBG` under "pcacti" to compile it in debug and un-optimized mode. Then invoke your profiler, for instance, run `valgrind --tool=callgrind ./cacti -infile ../model-pcacti/ref.xml`, and then `callgrind-annotate` in directory "pcacti".