# Time Series Characteristics

[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg?style=popout-square)](LICENSE.txt)

Time series characteristics are features calculated from time series.
This repository contains a wide range of time series characteristics that were collected from related work.
We sorted and grouped them according to various properties into the following four main groups and their subgroups:

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

Here is a detailed table with all features within the above groups:

| Group                 | Subgroup                 | Feature                                | Descripion                                                                                                                                             |
|-----------------------|--------------------------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Distributional**    | Dispersion               | `kurtosis`                             | measure of tailedness                                                                                                                                  |
|                       |                          | `skewness`                             | measure of asymmetry                                                                                                                                   |
|                       |                          | `shift`                                | mean minus the median of those values that are smaller than the mean                                                                                   |
|                       | Blockwise Dispersion     | `lumpiness`                            | variance of the variances of blocks                                                                                                                    |
|                       |                          | `stability`                            | variance of the mean of blocks                                                                                                                         |
|                       | Duplicates               | `normalized_duplicates_max`            | number of duplicates that have the maximum value of the data                                                                                           |
|                       |                          | `normalized_duplicates_min`            | number of duplicates that have the minimum value of the data                                                                                           |
|                       |                          | `percentage_of_reoccurring_datapoints` | number of unique duplicates compared to the number of unique values                                                                                    |
|                       |                          | `percentage_of_reoccurring_values`     | number of duplicates compared to the length of the data                                                                                                |
|                       |                          | `percentage_of_unique_values`          | number of unique values compared to the length of the data                                                                                             |
|                       | Distribution             | `quantile`                             | threshold below which *x*% of the ordered values of the data are, giving a hint on the distribution                                                    |
|                       |                          | `ratio_beyond_r_sigma`                 | ratio of values that are more than a factor *r* · 𝜎 away from the mean                                                                                 |
|                       |                          | `ratio_large_standard_deviation`       | ratio between the standard deviation and the (max − min) range of the data (based on the "range rule of thumb")                                        |
| **Temporal**          | Dispersion               | `mean_abs_change`                      | average absolute difference of two consecutive values                                                                                                  |
|                       |                          | `mean_second_derivative_central`       | measure of the rate of the rate of change                                                                                                              |
|                       | Blockwise Dispersion     | `level_shift`                          | maximum difference in mean between consecutive blocks                                                                                                  |
|                       |                          | `variance_change`                      | maximum difference in variance between consecutive blocks                                                                                              |
|                       | Similarity               | `hurst`                                | measure of long-term memory of a time series, related to auto-correlation                                                                              |
|                       |                          | `autocorrelation`                      | correlation of a signal with a lagged version of itself                                                                                                |
|                       | Frequency                | `periodicity`                          | power (intensity) of specified frequencies in the signal (based on the periodogram)                                                                    |
|                       |                          | `agg_periodogram`                      | results of user-defined aggregation functions (e.g., fivenum) calculated on the periodogram                                                            |
|                       | Linearity                | `linear_trend_slope`                   | measure of linearity: slope                                                                                                                            |
|                       |                          | `linear_trend_rvalue2`                 | measure of linearity: *r* <sup>2</sup> (coefficient of determination)                                                                                  |
|                       |                          | `agg_linear_trend_slope`               | variance-aggregated slopes of blocks                                                                                                                   |
|                       |                          | `agg_linear_trend_rvalue2`             | mean-aggregated *r* <sup>2</sup> of blocks                                                                                                             |
|                       |                          | `c3`                                   | measure of non-linearity (originally from the physics domain)                                                                                          |
|                       |                          | `time_reversal_asymmetry_statistic`    | asymmetry of the time series if reversed, which can be a measure of non-linearity                                                                      |
| **Complexity**        | Entropy                  | `binned_entropy`                       | fast entropy estimation based on equidistant bins                                                                                                      |
|                       |                          | `kullback_leibler_score` (KL score)    | maximum difference of KL divergences between consecutive blocks, where the KL divergence is a measure of how two probability distributions differ      |
|                       |                          | `index_of_kullback_leibler_score`      | relative location where the maximum KL score was found                                                                                                 |
|                       | Miscellaneous Complexity | `cid_ce`                               | measure of complexity invariance                                                                                                                       |
|                       |                          | `permutation_analysis`                 | measure of complexity through permutation                                                                                                              |
|                       |                          | `swinging_door_compression_rate`       | compression ratio of the signal under a given error tolerance 𝜖                                                                                        |
|                       | Flatness                 | `normalized_crossing_points`           | number of times a time series crosses the mean line (based on fickleness)                                                                              |
|                       |                          | `normalized_above_mean`                | number of values that are higher than the mean                                                                                                         |
|                       |                          | `normalized_below_mean`                | number of values that are lower than the mean                                                                                                          |
|                       |                          | `normalized_longest_strike_above_mean` | relative length of the longest series of consecutive values above the mean                                                                             |
|                       |                          | `normalized_longest_strike_below_mean` | relative length of the longest series of consecutive values below the mean                                                                             |
|                       |                          | `flat_spots`                           | maximum run-length of values when divided into quantile-based bins                                                                                     |
|                       | Peaks                    | `normalized_number_peaks`              | number of peaks, where a peak of support *n* is defined as a value which is bigger than its *n* left and *n* right neighbors                           |
|                       |                          | `step_changes`                         | number of times the time series significantly shifts its value range                                                                                   |
| **Statistical Tests** |                          | `adf`                                  | augmented Dickey-Fuller (ADF) test for unit root presence                                                                                              |
|                       |                          | `kpss`                                 | Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity                                                                                         |



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
