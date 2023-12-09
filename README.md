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

To make PCACTI accept cache configuration input through command line arguments, please add the following lines to PCACTI's source code, file io.cc, function cacti_interface just before line 848 (`init_tech_params(g_ip->F_sz_um, false);`):

```
strcpy(g_ip->data_array_cell_tech_file,"xmls/devices/cmos_14nm_std.xml");
strcpy(g_ip->data_array_peri_tech_file,"xmls/devices/cmos_14nm_std.xml");
strcpy(g_ip->tag_array_cell_tech_file,"xmls/devices/cmos_14nm_std.xml");
strcpy(g_ip->tag_array_peri_tech_file,"xmls/devices/cmos_14nm_std.xml");
```

## How to run code
First, specify your settings in settings.cfg. For each line under "Input", specify the name of the cache configs to vary (as they appear in ref.xml) and their permissible values. For each line under "Output", specify the name of a output attribute you want to collect (as they appear in PCACTI's stdout). For each line under "Setup", specify the how the machine learning experiment is run, see comments in settings.cfg file.

Run `python train.py` to collect data, train model and evaluate model. The first time you run the program, data will be collected from PCACTI and written to data.csv, all combinations of cache config is explored. An errs.csv is also generated containing invalid cache configurations that we attempted to use. To trigger (re-)collection of data from PCACTI, you must delete "data.csv" or make sure it doesn't exist (or else the cached data in data.csv is used).

Currently, we implemented four models: Multi-layer Perceptron regressor and K-nearest neighbors, Support Vector Regressor and Random Forest. train.py also supports hyperparameter search. analyze_running_time.py analyzes and compares running time of cacti and models. batch_experiment.py runs a series of experiments in the random train-test-split setting.