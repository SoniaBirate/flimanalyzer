# FLIMAnalyzer

Python package for analysis of Fluoresence Lifetime Imaging Microscopy (FLIM) data.

## Requirements:

* Python 3.7 (or later)
* pandas
* numpy (1.18)
* matplotlib
* seaborn
* pytorch
* scikit-learn
* xlsxwriter
* wxpython (4.0.7)
* pypubsub
* python.app (for Mac OSX)

**Windows Installation:**
* Install VS Code

## Installation:

```
git clone https://github.com/uvaKCCI/flimanalyzer.git
cd flimanalyzer
conda env create -f environment.yml
conda activate flimenv # the name for your environment may differ
pip install -e .
```

This will create a Conda environment `flimenv` that contains all the Python packages required to run the FLIM Analyzer application.

**MacOS X Installation**

Run the general installation commands, activate the Conda environment and install the MacOS specific python.app package. Then patch the shebang of the flimanalyzer console script.

```
conda activate flimenv # the name for your environment may differ
conda install python.app
sed -i '' -e '1 s/^.*$/#!\/usr\/bin\/env python.app/' $(which flimanalyzer)
```

## Run the application

**From the command line**

On Windows, run this command
```
conda activate flimenv
set PREFECT__FLOWS__CHECKPOINTING=true 
flimanalyzer.exe
```

On Powershell, run this command
```
conda activate flimenv
$env:PREFECT__FLOWS__CHECKPOINTING='true'
flimanalyzer.exe
```

On Mac OSX and Linux, run this command
```
conda activate flimenv
export PREFECT__FLOWS__CHECKPOINTING=true 
flimanalyzer
```

**Parallel execution**

FLIMAanauzer uses [Prefect](https://www.prefect.io) and [Dask](https://www.dask.org) for parallel execution of tasks.For parallel execution on a *single node*, add these command line arguments:
```
export PREFECT__FLOWS__CHECKPOINTING=true 
flimanalyzer -e LocalDaskExecutor --execargs="scheduler=processes,num_workers=8" # or flimanalyzer.exe
```

As a general guideline, adjust `num_workers` to match the number of cpu cores in your system. 

