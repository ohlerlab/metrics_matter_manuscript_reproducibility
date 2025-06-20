[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15642298.svg)](https://doi.org/10.5281/zenodo.15642298)

# Shortcomings of Silhouette in Single-Cell Integration Benchmarking
This GitHub repository contains code, data, and environment configuration files for reproducing the analyses and figures of the manuscript: "Shortcomings of Silhouette in Single-Cell Integration Benchmarking.", a revised version of the preprint "Metrics Matter: Why We Need to Stop Using Silhouette in Single-Cell Benchmarking." [[1]](#1)

## Repository Structure
> Note: The directory structure has been reorganized for clarity. Some absolute and relative paths in scripts may need adjustment, but can be inferred from the structure and naming conventions below.

```
metrics_chapter/
├── configs/         # Conda YAML environment files
├── data/            # Contains both simulated data and sources of original data
│   ├── original/    # Sources of original data
│   ├── simulated/   # Simulated data
│   └── simulated_2d/# 2D simulated data
├── embeddings/      # Output embeddings from integration methods
├── evaluation/      # Metric scores for all analyses
└── scripts/         # Analysis scripts
    ├── real_data/
    ├── real_data_hbca/
    ├── real_data_hlca/
    ├── real_data_minimal_example/
    ├── simulation/
    └── simulation_2d/
```

## Scenario to File Mapping
### Simulated 2D data
All analyses related to simulated 2D data contain "2D" in file names (Figure 1 and Extended Data Figure 2). 

### Simulated scRNA-seq Data
All analyses related to "Simulated scRNA-seq data" contain "simul*" and not "2D" in file names (Extended Data Figure 5).

### Real Data
#### NeurIPS data
- "NeurIPS data minimal example" (Minimal data subset of NeurIPS data set): Files containing "real_data_minimal_example" (Figure 2 and Extended Data Figure 6 and Supplementary Figure 1)
- "Full NeurIPS data": Files containing "real_data" but not "minimal_example" (Extended Data Figure 3 and Supplementary Figure 4)

#### HLCA data
All analyses related to the HLCA data contain "hlca*" in file names (Figure 2 and Extended Data Figure 6 and Supplementary Figure 2).

#### HBCA data
All analyses related to the HBCA data contain "hbca*" in file names (Extended Data Figure 4 and Supplementary Figure 3).

### Supplementary Analyses
Analyses and figures related to Supplementary Note 3 can be found in files containing "*optimization_clustering_resolution.ipynb".

## Custom Metric Implementations
### Batch Removal Adapted Silhouette (BRAS) Metric
We implement the BRAS metric in the ```scripts/custom_silhouette_functions.ipynb``` Jupyter notebook and made a scaleable and readily installable version availabe as part of the [scib-metrics package](https://github.com/yoseflab/scib-metrics) as of version 0.5.5.

### CiLISI
Find and example custom CiLISI implementations in `scripts/simulation/Evaluate_simulation.ipynb`. Identical implementations are available as part of the evaulation scripts for the other data sets.

## References
<a id="1">[1]</a>
Rautenstrauch, P. & Ohler, U. (2025) [Metrics Matter: Why We Need to Stop Using Silhouette in Single-Cell Benchmarking](https://doi.org/10.1101/2025.01.21.634098). bioRxiv DOI: 10.1101/2025.01.21.634098

---

For questions or issues, please open an issue on GitHub or contact me via e-mail.
