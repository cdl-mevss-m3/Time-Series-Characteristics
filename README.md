# Time Series Characteristics

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=popout-square)](LICENSE.txt)

Time series characteristics are features calculated from time series.
This repository contains a wide range of time series characteristics that were collected from related work.
We sorted and grouped them according to various properties into the following four groups:

- **Distributional Features**
    - Measures of Dispersion
    - Measures of block-wise Dispersion
    - Measures on the Number of Duplicates
    - Measures on the Distribution
- **Temporal Features**
    - Measures of Temporal Dispersion
    - Measures of block-wise Temporal Dispersion
    - Measures of Temporal Similarity
    - Measures in the Frequency Spectrum
    - Measures of Linearity and Trends
- **Complexity Features**
    - Measures of Entropy
    - Measures of (miscellaneous) Complexity and Permutation
    - Measures of Flatness
    - Measures of Peaks & Peakiness
- **Statistical Tests** (Stationarity and Unit Roots)
    - Augmented Dickey-Fuller (ADF)
    - Kwiatkowski-Phillips-Schmidt-Shin (KPSS)

## Implementation

| :warning: Please note that this is just a prototype without any major optimizations, so run-time performance can still be improved drastically. |
|-|

The implementation of the time series characteristics is in [`time_series_characteristics.py`](tsc/time_series_characteristics.py).

## Jupyter Notebook

The Jupyter notebook [`feature_groups.ipynb`](tsc/feature_groups.ipynb) provides more details on the time series characteristic groups and subgroups, including information on why we decided to drop some features, information on normalization, from where the features originate, etc.

To quickly gain some insights, an evaluted notebook is provided as an HTML file [`feature_groups.html`](tsc/feature_groups.html).

### Running the Jupyter Notebook

You need some additional dependencies for running the Jupyter notebook:

```
conda install jupyterlab ipywidgets
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Reference

If this work is valuable to your work, please reference this repository or our paper _(which is currently under review - stay tuned, link follows)_.
