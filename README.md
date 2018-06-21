## Autonomous construction of block structures with a swarm of quadcopters

This repository contains part of the code for my Bachelor Thesis project at the Department of Data Science and Knowledge Engineering at Maastricht University. The code implements a high-level simulation for autonomous construction using quadcopters. The `res` folder contains files with representations for a large number of different structures that can be built using the simulator.

## Installation

The code requires Python 3. All dependencies can be installed from the included `requirements.txt` file. Note that the code contains some absolute file paths referring to my personal computer. Therefore the experiment utility (`experiments.py`) should be started with the correct command line flags for loading and saving data.

## Usage

For launching a simple GUI to start a single run of simulated construction, `main.py` can be called from the command line or in an IDE.

The file `experiments.py` launches a command line interface for configuring and running experiments. The following is the help message describing its correct usage. 

```commandline
usage: experiments.py [-h] [-l LOAD_DIRECTORY] [-s SAVE_DIRECTORY]
                      [--skip-existing]
                      map_name

Run and save results of simulated construction with quadcopters.

positional arguments:
  map_name              Name of the map/structure to build.

optional arguments:
  -h, --help            show this help message and exit
  -l LOAD_DIRECTORY, --load-directory LOAD_DIRECTORY
                        Either a number (0-2) specifying one of three pre-
                        defined paths to load maps from or an absolute file
                        path.
  -s SAVE_DIRECTORY, --save-directory SAVE_DIRECTORY
                        Either a number (0, 1) specifying one of two pre-
                        defined paths to save results to or an absolute file
                        path.
  --skip-existing       Specifies whether existing files should be overwritten
                        if experiments are repeated. The default is to
                        overwrite them.
```

The functions in `analysis.py` were used for analysis of the data gathered from the experiments. It is currently undocumented and *very* messy.