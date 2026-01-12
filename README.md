# 1. Quick start guide

## 1.1. Introduction

This is a Harm Reduction Cost-Benefit Analysis tool. It is developed as a collaboration between the Burnet Institute and the Australian National University.

This repository contains all the code required to run the tool (https://git01.burnet.edu.au/modelling/act-harm-reduction).

If you have any questions, please email tidhar.tom@gmail.com

## 1.2. Requirements

This project requires Python 3.12 which can be downloaded directly from the Python website (https://www.python.org/downloads/release/python-31212/)

The package requirements are outlined in the [requirements](requirements.txt) file (requirements.txt) in this repository.


# 2. File structure

## 2.1. Model spreadsheets

The Harm Reduction model is represented by two spreadsheets (model and databook) which are located in the GitLab repository.

1. [model](model_info_20250319.xlsx): This outlines the structure of the Directed Acyclic Graphs (DAGs) to be modelled. Each DAG consists of nodes and arcs.
2. [databook](databook_20250924_standard_github.xlsx): This outlines specific data and parameter estimates, and is separated into tabs:
   - _probabilities_: The Conditional Probability Tables (CPTs) for each node -i.e. the probability of an outcome occurring given a set of conditions.
   - _evidence_: Data about the probability of a given outcome being observed.
   - _population_size_: Size estimates for the different populations in the model.
   - _costs_: Each outcome can have an associated unit cost, or setup cost (one-off), or both.
   - _max_coverage_: The maximum coverage of the different interventions.

## 2.2. Scripts

The model is seperated into several scripts for convenience.
- [main.py](main.py): The main script to run, which contains all the key folder references and parameters to be selected and changed.
- [graph.py](graph.py): Contains the functionality for creating and using DAGs.
- [plotting.py](plotting.py): Contains the functionality for plotting results.
- [utils.py](utils.py): Contains additional convenience and utility functions.
- [constants.py](constants.py): Contains constants which are referenced in the model, such as annual discounting rate.


# 3. How to use

This section describes how to run and use the model.

## 3.1. Specifying parameters

There are several key folders and parameters which need to specified in the [main.py](main.py) script:
- _data_folder_: This must be set to the folder location of the model and databook spreadsheet files. The model will look here for the files.
- _desktop_folder_: This is the location where the model output will be saved. Defaults to a "Results_..." folder on the desktop.
- _model_file_: This is the name or pattern for the model spreadsheet. If a pattern is used, the model will select the last file alphabetically which fits that pattern.
- _databook_file_: This is the name or pattern for the databook spreadsheet. If a pattern is used, the databook will select the last file alphabetically which fits that pattern.
- _n_samples_: The number of samples to be used for each run. It is recommended to set this to a minimum of 100 for sufficient accuracy.
- _seed_: The seed to be used for the model. Ensures reproducibility of results.

## 3.2. Running the model

When these parameters have been specified correctly, the user must run the [main.py](main.py) script.

Once the script has successfully finished running, the results will be saved to the output folder specified in _desktop_folder_ (see Section 3.1 above).

# 4. Results

The model produces several key results

## 4.1. Spreadsheets

The results (including best estimates as well as summarised sensitivity results) are saved in Excel spreadsheets.

## 4.2. Plots

The results are visualised using various plots, automatically produced and saved in a unique folder within the results folder.

## 4.3. Run info

A text file containing information about the model run, as well as copies of the model and databook spreadsheets used for the run.


