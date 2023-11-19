# Code Directory README

## Overview
This README provides guidance on the organization and usage of the `code` directory. 
This directory contains folders for each dataset, and within these, you will find Python scripts for different versions of the experiments.

## Directory Structure
- `code/`
  - `dataset1/`
    - `k_means_uniform.py`
    - `k_means++.py`
    - `hac_single.py`
    - `hac_average.py`
  - `dataset2/`
    - `...` (similar structure for each dataset)
-`data/
  -`dataset1.csv`
  -`dataset2.csv`

## Description of Python Scripts
- `k_mean.py`: Runs the K-Means clustering experiment.
- `k_mean++.py`: Executes the K-Means++ clustering experiment.
- `hac_single.py`: Performs Hierarchical Agglomerative Clustering (HAC) with single linkage.
- `hac_average.py`: Implements HAC with average linkage.

## Running the Experiments
1. Navigate to the top level in the directory.
2. Run the desired Python file. For example: 
      python3 code/dataset1/*.py # To run all files in dataset1
3. Each script will create a folder inside the `visuals` directory (`visuals` is created with any of the scripts).
4. Inside the `visuals` folder, you will find graphs and visualizations specific to that experiment.

## Notes
- Ensure Python is installed and properly configured on your system.
- Each script is designed to work with its corresponding dataset in the same folder.
- The `visuals` folder is generated dynamically; make sure you have the necessary permissions in the directory.

