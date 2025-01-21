# Metrics Matter: Why We Need to Stop Using Silhouette in Single-Cell Benchmarking
This GitHub repository contains code, data, and environment configuration files for reproducing the analyses and figures of the manuscript: "Metrics Matter: Why We Need to Stop Using Silhouette in Single-Cell Benchmarking."

## Repository Structure
Due to reorganizing the scripts in the directory structure for improved readability, absolute and relative paths may be incorrect but can be inferred from the directory structure and naming conventions.

## Scenario to File Mapping
### Simulated Data
All analyses related to "Simulated data" contain "simul*" in file names (main figure).

### Real Data
- "NeurIPS data minimal example" (Minimal data subset of NeurIPS data set): Files containing "real_data_minimal_example" (main figure)
- "Full NeurIPS data": Files containing "real_data" (supplementary figure)

## Custom Metric Implementations

### Batch Removal Adapted Silhouette (BRAS) Metric
We implement the BRAS metric in the ```scripts/custom_silhouette_functions.ipynb``` Jupyter notebook.

### CiLISI
Find custom CiLISI implementations (identical)  in the following scripts:
- ```scripts/simulation/Evaluate_simulation.ipynb```
- ```scripts/real_data_minimal_example/Evaluate_real_data_minimal_example.ipynb```
- ```scripts/real_data/Evaluate_real_data.ipynb```

## Supplementary Analyses
Analyses and figures related to Supplementary Note 1 can be found in files containing "*optimization_clustering_resolution.ipynb".

## Getting Started
YAML files for conda environments are in the ```config``` directory.