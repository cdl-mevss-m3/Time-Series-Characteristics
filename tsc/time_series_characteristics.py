import collections
import warnings
from math import sqrt
from typing import Union, Tuple

import nolds
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from arch import unitroot
from numpy.linalg import LinAlgError
from pandas.tseries.frequencies import to_offset
from scipy import signal
from sklearn.neighbors import KernelDensity
from tsfresh.feature_extraction import feature_calculators as fc

import periodicity as per
import swinging_door as swd
import util


# noinspection PyMethodMayBeStatic
class TimeSeriesCharacteristics:
    """
    The base class for calculating time series characteristics. Each function has
    the time series ``data`` as first parameter. The characteristics are designed
    to work on standardized (zero mean, unit variance) data, so ensure that all
    time series are standardized before calling the functions of this class.
    """
    
    def __init__(self, block_padding_limit: float = None):
        """
        Creates a new ``TimeSeriesCharacteristics``.
        
        :param block_padding_limit: If not None, specifies the limit in % of the data
            size, which the last block shall not exceed when calculating block-wise
            metrics. This takes effect in case the data length is not evenly divisible
            by the block size and thus the last block is smaller. If this last block
            is larger than this limit, an exception is raised (default: None)
            Example:
                data length = 100, block size = 60, block_padding_limit = 0.2 = 20%
                last block size = 40 = 40% of data length
                40% > 20% ---> raise exception
        """
        self.block_padding_limit = block_padding_limit
    
    ##########################################################################
    # helper functions
    ##########################################################################
    
    @staticmethod
    def _ensure_ndarray(data):
        if isinstance(data, pd.Series):
            return data.values
        if isinstance(data, np.ndarray):
            return data
        return np.array(data, copy=False)
    
    @staticmethod
    def _check_block_padding(data_size, block_size, limit):
        """
        Block-based metrics usually run into the problem that the last block is smaller, if the data length is
        not evenly divisible by the block size. This function raises an exception if the last block size is
        larger than x % of the overall data size.

        :param data_size: The length of the data
        :param block_size: The length of the block
        :param limit: The limit in % of the data size, which the last block shall not exceed
        :return The padding ratio, i.e. the size of the remaining, last block, in % of the overall data size
        """
        padding = (data_size % block_size) / data_size
        if padding > limit:
            raise Exception("block size {:d} is not evenly divisible and would ignore the last block of size {:d} % {:d} = {:d}, "
                            "which is too large to be ignored ({:.1%}) - change the block size or increase the limit (current limit: {:.1%})"
                            .format(block_size, data_size, block_size, data_size % block_size, padding, limit))
        return padding
    
    def _block_metrics(self, data, block_size: int):
        """
        Computes stability, lumpiness, level shift, and variance change in one pass
        :return: A dictionary with keys index and value
        """
        if self.block_padding_limit is not None:
            TimeSeriesCharacteristics._check_block_padding(len(data), block_size, self.block_padding_limit)
        
        means, variances = [], []
        for i in range(0, data.shape[0], block_size):
            block = data[i:i + block_size]
            means.append(block.mean())
            variances.append(block.var(ddof=1))
        
        stability_ = np.var(means)
        lumpiness_ = np.var(variances)
        level_shift_ = np.nan if len(means) <= 1 else max(np.abs(np.diff(means)))
        variance_change_ = np.nan if len(variances) <= 1 else max(np.abs(np.diff(variances)))
        
        return dict(stability=stability_,
                    lumpiness=lumpiness_,
                    level_shift=level_shift_,
                    variance_change=variance_change_)
    
    def _kullback_leibler_core(self, data: Union[np.ndarray, pd.Series], block_size: int,
                               interval: Union[str, Tuple[int, int]] = "infer", resolution: int = 100):
        """
        Computes the Kullback-Leibler score, which is the difference of
        Kullback-Leibler divergences of consecutive blocks.
        The distribution within a block is estimated with a Gaussian KDE.
        The maximum difference between Kullback-Leibler divergences is returned.

        Invented by Hyndman et al. https://doi.org/10.1109/ICDMW.2015.104

        :param data: The time series as a one-dimensional set of data points
        :param block_size: The number of data points per block
        :param interval: The (min, max) interval of the data, on which the distribution shall be estimated.
        If you choose "infer", the minimum and maximum of the data is inferred automatically. (default: "infer")
        :param resolution: The resolution of the density estimation (default: 100)
        :return: A dictionary with keys index and value
        """
        # min = -inf, max = inf
        if self.block_padding_limit is not None:
            TimeSeriesCharacteristics._check_block_padding(len(data), block_size, self.block_padding_limit)
        
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        
        # the value range, onto which we estimate the distribution
        if interval == "infer":
            x_space = np.linspace(np.min(data), np.max(data), resolution)
        else:
            x_space = np.linspace(*interval, resolution)
        x_space = x_space.reshape(-1, 1)
        
        # estimate the kde bandwidth parameter with Silverman's rule of thumb over the entire data space
        bw = 0.9 * min(np.std(data), sp.stats.iqr(data) / 1.34) * (len(data) ** (- 1 / 5))
        bw = max(0.05, bw)  # ... avoid too narrow bandwidths
        
        kls, probs_pre = [], None
        for i in range(0, data.shape[0], block_size):
            block = data[i:i + block_size].reshape(-1, 1)
            
            # ignore the last block if its not a full one any more
            if len(block) != block_size:
                break
            
            # kde of current block
            kde = KernelDensity(bandwidth=bw, kernel="gaussian", metric="euclidean")
            kde.fit(block)
            probs = np.exp(kde.score_samples(x_space))
            probs[probs < 1E-6] = 1E-6  # avoid divisions by zero
            
            # kl divergence between consecutive blocks
            if i > 0:
                kls.append(sp.stats.entropy(probs_pre, probs))
            
            probs_pre = probs
        
        # the maximum of kl divergence differences
        kls_diff = np.diff(kls)
        if len(kls_diff) == 0:
            return dict(index=np.nan, value=np.nan)
        
        kl_diff_max_index_ = np.argmax(kls_diff)
        
        return dict(index=kl_diff_max_index_ / len(kls_diff),
                    value=kls_diff[kl_diff_max_index_])
    
    ##########################################################################
    # 1. Distributional Features
    ##########################################################################
    
    ##########################################################################
    # 1.1. Distributional Dispersion Features
    ##########################################################################
    
    @util.copy_doc_from(fc.kurtosis)
    def kurtosis(self, data):
        # min = -3, max = inf
        return fc.kurtosis(data)
    
    @util.copy_doc_from(fc.skewness)
    def skewness(self, data):
        # min = -inf, max = inf
        return fc.skewness(data)
    
    def shift(self, data):
        """
        Returns the mean minus the median of those values that are smaller than the mean.
        """
        # min = -inf, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        
        mean_ = np.mean(data)
        subset_ = data[data < mean_]
        
        shift_ = np.mean(subset_) - np.median(subset_)
        return shift_
    
    ##########################################################################
    # 1.2. Distributional Dispersion Blockwise Features
    ##########################################################################
    
    def lumpiness(self, data, block_size: int):
        """
        Returns the variance of the variances of all (non-overlapping) blocks of size ``block_size``.
        """
        # min = 0, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        lumpiness_ = self._block_metrics(data, block_size)["lumpiness"]
        return lumpiness_
    
    def stability(self, data, block_size: int):
        """
        Returns the variance of the means of all (non-overlapping) blocks of size ``block_size``.
        """
        # min = 0, max = 1 (for z-norm data)
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        stability_ = self._block_metrics(data, block_size)["stability"]
        return stability_
    
    ##########################################################################
    # 1.3. Distributional Duplicates Features
    ##########################################################################
    
    def normalized_duplicates_max(self, data):
        """
        Returns ``x / len(data)`` where ``x`` is the number of duplicates that
        have the maximum value of ``data``. If there are no duplicates, i.e.,
        the maximum value occurs only once, 0 is returned.
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        count = np.sum(data == np.max(data))
        if count > 1:
            denom = len(data)
            return count / denom if denom != 0 else np.nan
        return 0
    
    def normalized_duplicates_min(self, data):
        """
        Returns ``x / len(data)`` where ``x`` is the number of duplicates that
        have the minimum value of ``data``. If there are no duplicates, i.e.,
        the minimum value occurs only once, 0 is returned.
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        count = np.sum(data == np.min(data))
        if count > 1:
            denom = len(data)
            return count / denom if denom != 0 else np.nan
        return 0
    
    @util.copy_doc_from(fc.percentage_of_reoccurring_datapoints_to_all_datapoints)
    def percentage_of_reoccurring_datapoints(self, data):
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return fc.percentage_of_reoccurring_datapoints_to_all_datapoints(data)
    
    @util.copy_doc_from(fc.percentage_of_reoccurring_values_to_all_values)
    def percentage_of_reoccurring_values(self, data):
        # min = 0, max = 1
        return fc.percentage_of_reoccurring_values_to_all_values(data)
    
    @util.copy_doc_from(fc.ratio_value_number_to_time_series_length)
    def percentage_of_unique_values(self, data):
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return fc.ratio_value_number_to_time_series_length(data)
    
    ##########################################################################
    # 1.4. Distributional Distribution Features
    ##########################################################################
    
    @util.copy_doc_from(fc.quantile)
    def quantile(self, data, q: float):
        # min = -inf, max = inf
        return fc.quantile(data, q)
    
    @util.copy_doc_from(fc.ratio_beyond_r_sigma)
    def ratio_beyond_r_sigma(self, data, r):
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return fc.ratio_beyond_r_sigma(data, r)
    
    def ratio_large_standard_deviation(self, data):
        """
        Returns the ratio between the standard deviation and the ``(max − min)`` range
        of the data, based on the range rule of thumb.
        """
        # min = 0, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        denom = np.max(data) - np.min(data)
        return np.std(data) / denom if denom != 0 else np.nan
    
    ##########################################################################
    # 2. Temporal Features
    ##########################################################################
    
    ##########################################################################
    # 2.1. Temporal Dispersion Features
    ##########################################################################
    
    @util.copy_doc_from(fc.mean_abs_change)
    def mean_abs_change(self, data):
        # min = 0, max = 2 * sqrt((m ** 2 + m + 1 / 4) / (m ** 2 + m))
        # with m = ceil(N / 2) and with N = data length (for z-norm data)
        # range normalized to [0, 1] assuming z-normed data
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        mac = fc.mean_abs_change(data)
        m = len(data) // 2
        denom = 2 * sqrt((m ** 2 + m + 1 / 4) / (m ** 2 + m))
        return mac / denom
    
    @util.copy_doc_from(fc.mean_second_derivative_central)
    def mean_second_derivative_central(self, data):
        # min = -inf, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return fc.mean_second_derivative_central(data)
    
    ##########################################################################
    # 2.2. Temporal Dispersion Blockwise Features
    ##########################################################################
    
    def level_shift(self, data, block_size: int):
        """
        Returns the maximum difference in mean between consecutive (non-overlapping) blocks of size ``block_size``.
        """
        # min = 0, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        level_shift_ = self._block_metrics(data, block_size)["level_shift"]
        return level_shift_
    
    def variance_change(self, data, block_size: int):
        """
        Returns the maximum difference in variance between consecutive (non-overlapping) blocks of size ``block_size``.
        """
        # min = 0, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        variance_change_ = self._block_metrics(data, block_size)["variance_change"]
        return variance_change_
    
    ##########################################################################
    # 2.3. Temporal Similarity Features
    ##########################################################################
    
    @util.copy_doc_from(nolds.hurst_rs)
    def hurst(self, data, **kwargs):
        # min = -inf, max = inf (should be between 0 and 1)
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return nolds.hurst_rs(data, **kwargs)
    
    @util.copy_doc_from(fc.autocorrelation)
    def autocorrelation(self, data, lag: int):
        # min = -1, max = 1
        # range normalized to [0, 1]
        return (fc.autocorrelation(data, lag) + 1) / 2
    
    ##########################################################################
    # 2.4. Temporal Frequency Features
    ##########################################################################
    
    @util.copy_doc_from(per.periodicity)
    def periodicity(self, data, periods=None, dt_min=None, replace_nan=0):
        # min = 0, max = inf
        if periods is None:
            periods = per.PERIODS
        result = per.periodicity(data, periods=periods, dt_min=dt_min)
        return result if replace_nan is None else result.fillna(replace_nan)
    
    def agg_periodogram(self, data, funcs, lowest_freq=None, highest_freq=None, dt_min=None):
        """
        Returns a list of tuples of aggregated periodogram power values. The first entry of
        a tuple is the name of applied function, the second is the calculated value.

        :param data: The time series.
        :param funcs: A list of numpy function strings or tuples. For a tuple, the first entry must be the
            numpy function string and the second entry must be a dict containing keyword arguments.
        :param lowest_freq: The lowest frequency to consider. Lower frequencies are discarded.
        :param highest_freq: The highest frequency to consider. Higher frequencies are discarded.
        :param dt_min: The time interval between values of ``data`` in minutes. If None, ``data`` must have a
            DateTimeIndex with a set frequency (e.g., via ``data = data.asfreq("1min")``) so the time interval
            can be inferred (default: None = infer time interval from ``data``).
        :return: A list of tuples of aggregated periodogram power values.
        """
        # min = 0, max = inf
        
        # time interval in minutes
        if dt_min is None:
            dt_min = pd.to_timedelta(to_offset(data.index.freq)).total_seconds() / 60
        
        fxx, pxx = signal.periodogram(data.values, fs=dt_min, return_onesided=True, detrend=None, scaling="spectrum")
        # skip the offset
        fxx = fxx[1:]
        pxx = pxx[1:]
        
        if lowest_freq is not None and highest_freq is not None:
            assert lowest_freq < highest_freq
            indices = np.argwhere((fxx >= lowest_freq) & (fxx <= highest_freq)).flatten()
            pxx = pxx[indices]
        elif lowest_freq is not None:
            indices = np.argwhere(fxx >= lowest_freq).flatten()
            pxx = pxx[indices]
        elif highest_freq is not None:
            indices = np.argwhere(fxx <= highest_freq).flatten()
            pxx = pxx[indices]
        
        result = []
        for f in funcs:
            if isinstance(f, str):
                method = getattr(np, f)
                result.append((f, method(pxx)))
            else:
                f, params = f
                method = getattr(np, f)
                params_str = "".join([f"__{k}_{v}" for k, v in params.items()])
                result.append((f"{f}{params_str}", method(pxx, **params)))
        return result
    
    ##########################################################################
    # 2.5. Temporal Linearity Features
    ##########################################################################
    
    @util.copy_doc_from(fc.linear_trend)
    def linear_trend_slope(self, data):
        """
        This method only returns the slope, i.e., param=[{"attr": "slope"}].
        """
        # min = -inf, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        # return value is a list where each list entry corresponds to one attribute; since we only
        # have one attribute, this list only has one entry; this entry is a tuple where the first
        # part is the attribute name ('attr_"slope"') and the second is the actual value
        return fc.linear_trend(data, param=[{"attr": "slope"}])[0][1]
    
    @util.copy_doc_from(fc.linear_trend)
    def linear_trend_rvalue2(self, data):
        """
        This method only returns the squared rvalue (= coefficient of determination),
        i.e., param=[{"attr": "rvalue"}].
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        # return value is a list where each list entry corresponds to one attribute; since we only
        # have one attribute, this list only has one entry; this entry is a tuple where the first
        # part is the attribute name ('attr_"rvalue"') and the second is the actual value
        rvalue = fc.linear_trend(data, param=[{"attr": "rvalue"}])[0][1]
        return rvalue ** 2
    
    @util.copy_doc_from(fc.agg_linear_trend)
    def agg_linear_trend_slope(self, data, block_sizes):
        """
        This method only returns the variance-aggregated slopes,
        i.e., param=[{..., "f_agg": "var", "attr": "slope"}].
        """
        # min = -inf, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        param = [{"f_agg": "var", "attr": "slope", "chunk_len": b} for b in block_sizes]
        return fc.agg_linear_trend(data, param)
    
    @util.copy_doc_from(fc.agg_linear_trend)
    def agg_linear_trend_rvalue2(self, data, block_sizes):
        """
        This method only returns the mean-aggregated squared rvalues (= coefficient of determination),
        i.e., param=[{..., "f_agg": "mean", "attr": "rvalue"}].
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        param = [{"f_agg": "mean", "attr": "rvalue", "chunk_len": b} for b in block_sizes]
        result = fc.agg_linear_trend(data, param)
        return [(key, val ** 2) for key, val in result]
    
    @util.copy_doc_from(fc.c3)
    def c3(self, data, lag):
        # min = -inf, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return fc.c3(data, lag)
    
    @util.copy_doc_from(fc.time_reversal_asymmetry_statistic)
    def time_reversal_asymmetry_statistic(self, data, lag: int):
        # min = -inf, max = inf
        return fc.time_reversal_asymmetry_statistic(data, lag)
    
    ##########################################################################
    # 3. Complexity Features
    ##########################################################################
    
    ##########################################################################
    # 3.1. Complexity Entropy Features
    ##########################################################################
    
    @util.copy_doc_from(fc.binned_entropy)
    def binned_entropy(self, data, max_bins: int):
        # min = 0, max = np.log(max_bins)
        # normalized to [0, 1]
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        denom = np.log(max_bins)
        return fc.binned_entropy(data, max_bins) / denom
    
    def kullback_leibler_score(self, data: Union[np.ndarray, pd.Series], block_size: int,
                               interval: Union[str, Tuple[int, int]] = "infer", resolution: int = 100):
        """
        Computes the Kullback-Leibler score, which is the difference of
        Kullback-Leibler divergences of consecutive blocks.
        The distribution within a block is estimated with a Gaussian KDE.
        The maximum difference between Kullback-Leibler divergences is returned.

        Invented by Hyndman et al. https://doi.org/10.1109/ICDMW.2015.104

        :param data: The time series as a one-dimensional set of data points
        :param block_size: The number of data points per block
        :param interval: The (min, max) interval of the data, on which the distribution shall be estimated.
        If you choose "infer", the minimum and maximum of the data is inferred automatically. (default: "infer")
        :param resolution: The resolution of the density estimation (default: 100)
        :return: The Kullback-Leibler score
        """
        # min = -inf, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return self._kullback_leibler_core(data, block_size, interval, resolution)["value"]
    
    def index_of_kullback_leibler_score(self, data: Union[np.ndarray, pd.Series], block_size: int,
                                        interval: Union[str, Tuple[int, int]] = "infer", resolution: int = 100):
        """
        Computes the index of the Kullback-Leibler score, which is the difference of
        Kullback-Leibler divergences of consecutive blocks.
        The distribution within a block is estimated with a Gaussian KDE.
        The maximum difference between Kullback-Leibler divergences is returned.

        Invented by Hyndman et al. https://doi.org/10.1109/ICDMW.2015.104

        :param data: The time series as a one-dimensional set of data points
        :param block_size: The number of data points per block
        :param interval: The (min, max) interval of the data, on which the distribution shall be estimated.
        If you choose "infer", the minimum and maximum of the data is inferred automatically. (default: "infer")
        :param resolution: The resolution of the density estimation (default: 100)
        :return: The (normalized) index of the Kullback-Leibler score
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return self._kullback_leibler_core(data, block_size, interval, resolution)["index"]
    
    ##########################################################################
    # 3.2. Complexity (Miscellaneous) Complexity Features
    ##########################################################################
    
    @util.copy_doc_from(fc.cid_ce)
    def cid_ce(self, data, normalize: bool = False):
        # min = 0, max = inf
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        return fc.cid_ce(data, normalize)
    
    def permutation_analysis(self, data):
        warnings.warn("not implemented due to NDA, returning 0")
        return 0
    
    def swinging_door_compression_rate(self, data, eps):
        """
        Returns the compression ratio of the data when using the swinging door compression algorithm:
        0 = no compression (number of compressed datapoints = number of original datapoints)
        1 = total compression (original datapoints can be represented with only 2 points: the start point + the end point)
        """
        # min = 0, max = 1
        
        # do not include error measures (such as in swd.compression_quality) because we cannot
        # return the error measures together with the compression rate as this would cause
        # problems with the distance measure in the clustering step
        
        x = range(len(data))
        y = TimeSeriesCharacteristics._ensure_ndarray(data)
        xc, yc = swd.swinging_door_sampling(x, y, eps)
        # the minimum compressed signal has a length of 2 (start point + end point), so subtract
        # 2 to get values in the range [0, 1]
        return 1 - ((len(yc) - 2) / (len(y) - 2))
    
    ##########################################################################
    # 3.3. Complexity Flatness Features
    ##########################################################################
    
    def normalized_crossing_points(self, data):
        """
        Returns the (normalized) number of times a time series crosses the mean line.
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        # the maximum number of segments above the mean is reached when the time series
        # starts above the mean and then continuously crosses the mean every timestamp;
        # this means that we get a segment above the mean every second timestamp
        # (up - below - up - below - up ...); we thus have at maximum len(data) / 2
        # segments above the mean + the additional one at the start of the time series
        # this one segment at the start is actually only relevant if the length of the
        # time series is odd; therefore, we could also write ceil(len(data) / 2)
        mean = np.mean(data)
        above = (data > mean).astype("int")
        count = ((np.diff(above) == 1).sum() + above[0])
        denom = (len(data) + 1) / 2  # equal to: np.math.ceil(len(data) / 2)
        return count / denom if denom != 0 else np.nan
    
    @util.copy_doc_from(fc.count_above_mean)
    def normalized_above_mean(self, data):
        """
        Returns ``r / (len(data) - 1)`` where ``r`` is the result of ``count_above_mean``.
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        denom = len(data) - 1
        return fc.count_above_mean(data) / denom if denom != 0 else np.nan
    
    @util.copy_doc_from(fc.count_below_mean)
    def normalized_below_mean(self, data):
        """
        Returns ``r / (len(data) - 1)`` where ``r`` is the result of ``count_below_mean``.
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        denom = len(data) - 1
        return fc.count_below_mean(data) / denom if denom != 0 else np.nan
    
    @util.copy_doc_from(fc.longest_strike_above_mean)
    def normalized_longest_strike_above_mean(self, data):
        """
        Returns ``r / len(data)`` where ``r`` is the result of ``longest_strike_above_mean``.
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        denom = len(data)
        return fc.longest_strike_above_mean(data) / denom if denom != 0 else np.nan
    
    @util.copy_doc_from(fc.longest_strike_below_mean)
    def normalized_longest_strike_below_mean(self, data):
        """
        Returns ``r / len(data)`` where ``r`` is the result of ``longest_strike_below_mean``.
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        denom = len(data)
        return fc.longest_strike_below_mean(data) / denom if denom != 0 else np.nan
    
    def flat_spots(self, data, n_intervals=10, mode="quantile", lower_bound=None, upper_bound=None):
        """
        Flat spots are computed by dividing the sample space of a time series into ten equal-sized intervals,
        and computing the maximum run length within any single interval. We can use ``n_intervals`` for the number
        of intervals and we can choose whether they should be equal-sized in the sense of equal value range
        (mode = "linear") or in the sense of equal number of datapoints in the intervals (mode = "quantile").
        We normalize the maximum run length of each interval with the length of the time series, i.e.,
        the sum of all interval max run lengths is at most 1
        :param data: The time series
        :param n_intervals: The number of bins into which the value space is divided
        :param mode: "linear" divides the value space equally, while "quantile"
            ensures that there is an equal number of data points per interval
        :param lower_bound: Enforce a lower bound on the value range
        :param upper_bound: Enforce an upper bound on the value range
        :return: The (normalized) maximum run length per interval
        """
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        max_run_lengths = dict()
        
        if mode == "quantile":
            bound_ids = np.linspace(0, 1, n_intervals + 1)
            intervals = np.quantile(data, q=bound_ids)
        elif mode == "linear":
            if lower_bound is None:
                lower_bound = np.min(data)
            if upper_bound is None:
                upper_bound = np.max(data)
            bound_ids = [i for i in range(n_intervals + 1)]
            intervals = np.linspace(lower_bound, upper_bound, n_intervals + 1)
        else:
            raise ValueError(f"unknown 'mode': {mode}")
        
        for j, (lower, upper, id_lower, id_upper) in enumerate(zip(intervals[:-1], intervals[1:], bound_ids[:-1], bound_ids[1:])):
            # to not miss any values, include everything below the lowest bound and everything above the upper most bound
            if j == 0:
                indices = np.argwhere(data < upper).flatten()
                bound_id = f"(-inf,{id_upper})"
            elif j == len(intervals) - 2:
                indices = np.argwhere(data >= lower).flatten()
                bound_id = f"[{id_lower},+inf)"
            else:
                indices = np.argwhere((data >= lower) & (data < upper)).flatten()
                bound_id = f"[{id_lower},{id_upper})"
            
            # if there are fewer than two values within this interval, there is no maximum run length (set to 0)
            if len(indices) < 2:
                max_run_lengths[bound_id] = 0
                continue
            
            i_diff = np.diff(indices)  # indices that follow each other (neighboring indices) have the value 1
            
            max_run_length = 0
            cur_run_length = 0
            for i in i_diff:
                if i == 1:  # we found another directly following index
                    cur_run_length += 1
                else:
                    if cur_run_length > max_run_length:
                        max_run_length = cur_run_length
                    cur_run_length = 0
            if cur_run_length > max_run_length:
                max_run_length = cur_run_length
            
            # since we work with diffs, the actual run length is max_run_length + 1
            # normalize against the total data length
            max_run_lengths[bound_id] = (max_run_length + 1) / len(data)
        
        assert len(max_run_lengths) == n_intervals
        return max_run_lengths
    
    ##########################################################################
    # 3.4. Complexity Peaks Features
    ##########################################################################
    
    @util.copy_doc_from(fc.number_peaks)
    def normalized_number_peaks(self, data, n: int):
        # min = 0, max = 1
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        # the maximum number of peaks is reached when the peaks are evenly distributed
        # within the time series signal and have the minimum required distance from each
        # other in order to be classified as peak, which is the distance "n"; this distance
        # must also be right at the start of the time series and at the end of the time
        # series; this means that there are "p" peaks in the time series and "p + 1" gaps
        # in between them (like: _^_^_^_ where "_" is the gap and "^" the peak); the length
        # of all gaps + the number of peaks must yield the total time series length;
        # mathematically, this can be expressed as "(p + 1) * n + p = len(data)" which can
        # rewritten to get the maximum number of peaks "p" like "p = (len(data) - n) / (n + 1)"
        denom = (len(data) - n) / (n + 1)
        return fc.number_peaks(data, n) / denom if denom != 0 else np.nan
    
    def step_changes(self, data, window_len):
        """
        A step change is counted whenever ``|y_i - mean{y_i-w...y_i-1}| > 2 * sigma{y_i-w...y_i-1}``,
        where, for every value ``y_i`` of the series, ``mean{y_i-w...y_i-1}`` and ``sigma{y_i-w...y_i-1}``
        are the mean and standard deviation in the preceding sliding window (of length ``w``) from point
        ``y_i-w`` to ``i-1``. Only full windows are considered, so the first ``window_len`` points are
        discarded. The result is normalized by the len of the sequences (- ``window_len``).
        :param data: The time series
        :param window_len: The length of the sliding window
        :return: The (normalized) number of step changes
        """
        # min = 0, max = 1
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
        rolling = data.rolling(window=window_len)
        mean = rolling.mean().values[window_len - 1:-1]
        std = rolling.std().values[window_len - 1:-1]
        y = data.values[window_len:]
        # Add tolerance of 1e-15 to avoid false positive due to numerical errors with std=0 and y-mean ~= 0
        count_step_changes = (np.abs(y - mean) > 2 * std + 1e-15).sum()
        denom = len(y)
        return count_step_changes / denom
    
    ##########################################################################
    # 4. Statistical Tests Features
    ##########################################################################
    
    @util.copy_doc_from(unitroot.ADF)
    def adf(self, data):
        # min = 0, max = 1
        # Augmented Dickey Fuller (ADF) Test
        # null hypothesis = data has a unit root (non-stationary)
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        try:
            return unitroot.ADF(data).pvalue
        # IndexError: for input containing only 0-valued elements or mostly 0-valued elements
        # AssertionError: for input containing only equal elements or mostly equal elements
        # LinAlgError: for input leading to a singular matrix for which a linear matrix equation cannot be solved
        except (IndexError, AssertionError, LinAlgError):
            return np.nan
    
    @util.copy_doc_from(unitroot.KPSS)
    def kpss(self, data):
        # min = 0, max = 1
        # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
        # null hypothesis = data is level or trend stationary
        data = TimeSeriesCharacteristics._ensure_ndarray(data)
        try:
            return unitroot.KPSS(data).pvalue
        # for input containing only equal elements or mostly equal elements
        except ValueError:
            return np.nan


class NormalizedTimeSeriesCharacteristics(TimeSeriesCharacteristics):
    """
    Derives from ``TimeSeriesCharacteristics`` and overrides all those functions that are not already
    normalized to the interval [0, 1].
    """
    
    def __init__(self, block_padding_limit=None,
                 norm_df=None, features=None, min_max_df=None, column_min=None, column_max=None, funcs_to_merge=None, param_separator="__"):
        """
        Creates a new ``NormalizedTimeSeriesCharacteristics``. There are three creation options:
         1) Specifying no parameters (parameterless constructor call): In this case, the created
            ``NormalizedTimeSeriesCharacteristics`` instance must be initialized manually using
            the method ``init`` before it can be used for normalization.
         2) Specifying ``norm_df`` (see method ``init`` for a full description)
         3) Specifying ``features``, ``min_max_df``, ``funcs_to_merge``, ``column_min``,
            ``column_max`` (see method ``init`` for a full description)
        """
        super().__init__(block_padding_limit)
        if norm_df is not None or features is not None:
            self.init(norm_df, features, min_max_df, column_min, column_max, funcs_to_merge, param_separator)
        else:
            self.norm_df = None
            self.column_min = None
            self.column_max = None
    
    ##########################################################################
    # helper functions
    ##########################################################################
    
    def init(self, norm_df=None, features=None, min_max_df=None, column_min=None, column_max=None, funcs_to_merge=None, param_separator="__"):
        """
        Initializes this ``NormalizedTimeSeriesCharacteristics`` instance to use the normalization
        values as specified by the ``norm_df`` DataFrame. This DataFrame must have two columns,
        namely a minimum column (per default, this is the first column in ``norm_df`` but can be
        explicitly specified with ``column_min``) and a maximum column (per default, this is the
        second column in ``norm_df`` but can be explicitly specified with ``column_max``). The index
        of ``norm_df`` must represent the corresponding functions which should be normalized using
        these two columns. The normalization is calculated via

        ``(func_result - norm_df.loc[i][column_min]) / (norm_df.loc[i][column_max] - norm_df.loc[i][column_min])``

        The index ``i`` of the DataFrame may look like:
         1) the name of the function

            All function return values will be normalized using the minimum and maximum of this
            single DataFrame entry.
         2) the name of the function in combination with the function parameters
            (cf. ``util.create_params_str(params)`` and ``util.create_func_result_identifier(func, params_str)``)

            This is for single-value return functions that use parameters
         3) the name of the function in combination with the function parameters and the function return identifier
            (cf. ``util.create_params_str(params)`` and ``util.create_func_result_identifier(func, params_str, key)``)

            All function return values will be normalized individually using the minimum and maximum
            values of the corresponding DataFrame entries.

        The ``norm_df`` DataFrame can be created using the method ``NormalizedTimeSeriesCharacteristics.create_norm_df``,
        but for convenience reasons, this can also be done internally. In this case, ``features``, ``min_max_df`` and
        ``funcs_to_merge`` must be specified instead of ``norm_df`` (optionally, also ``param_separator``, if the index
        entries of ``min_max_df``, i.e., the function names with parameters (and return identifiers), have a different
        parameter separator than the default double underscore). Moreover, ``column_min`` and ``column_max`` must now
        be mapped to the respective columns in ``min_max_df``.

        :param norm_df: The DataFrame containing the minimum (column 0) und maximum (column 1) values for
            different functions whose results should be normalized. If None, ``features``, ``min_max_df``
            ``column_min``, ``column_max`` and ``funcs_to_merge`` must be specified instead.
        :param features: A list of tuples where the first entry is the function object (this function must have one
            parameter in the first position that represents the data for which to calculate the feature) and
            the second entry is the dictionary containing the function's parameters (can be empty or None
            if the function does not have any parameters other than the required positional one). For
            convenience, it is also allowed to directly specify the function object instead of the tuple
            with the function object and parameters.
        :param min_max_df: A DataFrame where the index are function names (possibly with function parameters and
            function return identifiers (cf. ``util.create_params_str(params)`` and
            ``util.create_func_result_identifier(func, params_str, key)``) and the columns must contain at
            least a column holding minimum values (specified by ``column_min``) and a column holding the
            maximum values (specified by ``column_max``).
        :param column_min: If ``norm_df`` is specified, the name of the column of ``norm_df`` containing the
            minimum values (default: None, which means that the first column of ``norm_df`` will be used). If
            ``norm_df`` is not specified, i.e., if ``min_max_df`` is specified, then this is the name of the
            column of ``min_max_df`` containing the minimum values.
        :param column_max: If ``norm_df`` is specified, the name of the column of ``norm_df`` containing the
            maximum values (default: None, which means that the second column of ``norm_df`` will be used). If
            ``norm_df`` is not specified, i.e., if ``min_max_df`` is specified, then this is the name of the
            column of ``min_max_df`` containing the maximum values.
        :param funcs_to_merge: A list of function names which should be merged when those functions are called
            with different parameters.
        :param param_separator: The separator used for separating the function name and its parameters for the
            index entries in ``min_max_df``, which is a double underscore per default
            (cf. ``util.create_params_str(params)`` ---> ``param_separator``).
        """
        if norm_df is None:
            if column_min is None or column_max is None:
                raise ValueError("both 'column_min' and 'column_max' must be specified when not specifying 'norm_df'")
            norm_df = NormalizedTimeSeriesCharacteristics.create_norm_df(features, min_max_df, column_min, column_max,
                                                                         funcs_to_merge, param_separator)
        if norm_df.shape[1] != 2:
            raise ValueError("'norm_df' must have exactly 2 columns")
        self.column_min = column_min if column_min is not None else norm_df.columns[0]
        self.column_max = column_max if column_max is not None else norm_df.columns[1]
        self.norm_df = norm_df
        
        methods = util.get_overridden_methods(NormalizedTimeSeriesCharacteristics) - {"__init__"}
        norm_df_methods = {f if param_separator not in f else f.split(param_separator)[0] for f in norm_df.index}
        base_class_methods = norm_df_methods - methods
        if base_class_methods:
            base_class_methods_str = "\n\t" + "\n\t".join(sorted(base_class_methods))
            warnings.warn("'norm_df' contains {} methods which are already normalized per default (will be ignored):{}".format(
                len(base_class_methods), base_class_methods_str))
    
    @staticmethod
    def create_norm_df(features, min_max_df, column_min, column_max, funcs_to_merge, param_separator="__"):
        """
        Creates the ``norm_df`` DataFrame, which is a merged version of ``min_max_df``. Merging is based on the
        specified ``funcs_to_merge``, which is a list of function names. The index of ``min_max_df`` are also
        function names but possibly with function parameters and function return identifiers. For each function
        name in ``funcs_to_merge``, all matching rows of ``min_max_df`` are merged into a single row. Matching
        means that the index entries of ``min_max_df`` must start with the corresponding function name and, if
        the function has parameters (and return identifiers), must be separated by ``separator``, which is a
        double underscore per default (cf. ``util.create_params_str(params)`` ---> ``param_separator``).

        :param features: A list of tuples where the first entry is the function object (this function must have one
            parameter in the first position that represents the data for which to calculate the feature) and
            the second entry is the dictionary containing the function's parameters (can be empty or None
            if the function does not have any parameters other than the required positional one). For
            convenience, it is also allowed to directly specify the function object instead of the tuple
            with the function object and parameters.
        :param min_max_df: A DataFrame where the index are function names (possibly with function parameters and
            function return identifiers (cf. ``util.create_params_str(params)`` and
            ``util.create_func_result_identifier(func, params_str, key)``) and the columns must contain at
            least a column holding minimum values (specified by ``column_min``) and a column holding the
            maximum values (specified by ``column_max``).
        :param column_min: The name of the column of ``min_max_df`` containing the minimum values.
        :param column_max: The name of the column of ``min_max_df`` containing the maximum values.
        :param funcs_to_merge: A list of function names which should be merged when those functions are called
            with different parameters.
        :param param_separator: The separator used for separating the function name and its parameters for the
            index entries in ``min_max_df``, which is a double underscore per default
            (cf. ``util.create_params_str(params)`` ---> ``param_separator``).
        :return: ``norm_df`` DataFrame, which is the merged ``min_max_df``.
        """
        if funcs_to_merge is None:
            funcs_to_merge = []
        all_funcs = {f.__name__ if callable(f) else f[0].__name__ for f in features}
        funcs_to_keep = all_funcs - set(funcs_to_merge)
        
        index = []
        rows = []
        
        # merge these
        for func_name in funcs_to_merge:
            func_df = min_max_df.filter(regex=f"^{func_name}({param_separator}|$)", axis="index")
            index.append(func_name)
            rows.append([func_df.min().loc[column_min], func_df.max().loc[column_max]])
        norm_df = pd.DataFrame(rows, index=index, columns=[column_min, column_max])
        
        # keep these unchanged
        for func_name in funcs_to_keep:
            func_df = min_max_df.filter(regex=f"^{func_name}({param_separator}|$)", axis="index")[[column_min, column_max]]
            norm_df = norm_df.append(func_df)
        
        return norm_df
    
    def _normalize(self, func, data, minimum=None, maximum=None, **params):
        if self.norm_df is None:
            raise ValueError("this NormalizedTimeSeriesCharacteristics is not initialized: use method 'init'")
        
        result = func(data, **params)
        params_str = util.create_params_str(params)
        
        # for the name lookup in the norm_df.index, there are two cases:
        name = func.__name__
        # 1) the index contains the function name: this means we want to normalize the entire function result
        #    (both for a single-value result and multiple-value result) with the minimum and maximum specified
        #    in the row of this index entry ---> name_found = True
        # 2) the index does not contain the function name: this means that the index should contain multiple entries
        #    with the function name in combination with the function parameters and the function return identifiers
        #    since we now want to normalize each individual function result values with the minimum an maximum
        #    specified in the corresponding rows of the index entries ---> name_found = False
        name_found = name in self.norm_df.index
        
        if isinstance(result, collections.Iterable):
            normalized_result = []
            for key, val in util.transform_func_result(result):
                # for case 1), i.e., normalizing every result value with one minimum and maximum, we do not need to
                # do anything since the name lookup in the norm_df.index was already successful
                # for case 2), i.e., normalizing every result value individually, we now need to create the actual
                # lookup names for each individual result value
                key_name = name if name_found else util.create_func_result_identifier(func, params_str, key)
                
                key_minimum = self.norm_df.loc[key_name][self.column_min] if minimum is None else minimum
                key_maximum = self.norm_df.loc[key_name][self.column_max] if maximum is None else maximum
                denom = key_maximum - key_minimum
                
                if denom == 0:
                    warnings.warn(f"{key_name}: cannot normalize if minimum == maximum (returning original value)")
                    normalized_result.append((key, val))
                else:
                    normalized_result.append((key, (val - key_minimum) / denom))
            return normalized_result
        else:
            # of course, for the single result value (number), we also have to check whether the lookup in the
            # norm_df.index was successful; if not, this means that the index contains the function name in
            # combination with the function parameters (but no function return identifiers since we only have
            # a single result value)
            if not name_found:
                name = util.create_func_result_identifier(func, params_str)
            
            if minimum is None:
                minimum = self.norm_df.loc[name][self.column_min]
            if maximum is None:
                maximum = self.norm_df.loc[name][self.column_max]
            denom = maximum - minimum
            
            if denom == 0:
                warnings.warn(f"{name}: cannot normalize if minimum == maximum (returning original value)")
                return result
            
            return (result - minimum) / denom
    
    ##########################################################################
    # 1. Distributional Features
    ##########################################################################
    
    def kurtosis(self, data):
        # min = -3, max = inf
        return self._normalize(super().kurtosis, data, minimum=-3)
    
    def skewness(self, data):
        # min = -inf, max = inf
        return self._normalize(super().skewness, data)
    
    def shift(self, data):
        # min = -inf, max = inf
        return self._normalize(super().shift, data)
    
    def lumpiness(self, data, **params):
        # min = 0, max = inf
        return self._normalize(super().lumpiness, data, minimum=0, **params)
    
    def quantile(self, data, **params):
        # min = -inf, max = inf
        return self._normalize(super().quantile, data, **params)
    
    def ratio_large_standard_deviation(self, data):
        # min = 0, max = inf
        return self._normalize(super().ratio_large_standard_deviation, data, minimum=0)
    
    ##########################################################################
    # 2. Temporal Features
    ##########################################################################
    
    def mean_second_derivative_central(self, data):
        # min = -inf, max = inf
        return self._normalize(super().mean_second_derivative_central, data)
    
    def level_shift(self, data, **params):
        # min = 0, max = inf
        return self._normalize(super().level_shift, data, minimum=0, **params)
    
    def variance_change(self, data, **params):
        # min = 0, max = inf
        return self._normalize(super().variance_change, data, minimum=0, **params)
    
    @util.copy_doc_from(TimeSeriesCharacteristics.periodicity)
    def periodicity(self, data, **params):
        """
        The normalized version of ``TimeSeriesCharacteristics.periodicity`` does NOT
        return a pd.Series anymore. Instead, the series values are returned in a list
        containing 2-tuples, where the first entry of such a tuple is the index of the
        series and the second entry the corresponding value:

            old format = pd.Series (returned by ``TimeSeriesCharacteristics.periodicity``):
                index_1: value_1
                index_2: value_2
                ...
                index_n: value_n

            new format = List[Tuple] (returned by ``NormalizedTimeSeriesCharacteristics.periodicity``, i.e., this method):
                [(index_1, value_1),
                 (index_2, value_2),
                 ...
                 (index_n, value_n)]

        For a description of the parameters, refer to the documentation of
        ``TimeSeriesCharacteristics.periodicity`` (which, except for the return
        type, is exactly the same for this normalized version):
        """
        # min = 0, max = inf
        return self._normalize(super().periodicity, data, minimum=0, **params)
    
    def agg_periodogram(self, data, **params):
        # min = 0, max = inf
        return self._normalize(super().agg_periodogram, data, minimum=0, **params)
    
    def linear_trend_slope(self, data):
        # min = -inf, max = inf
        return self._normalize(super().linear_trend_slope, data)
    
    def agg_linear_trend_slope(self, data, **params):
        # min = -inf, max = inf
        return self._normalize(super().agg_linear_trend_slope, data, **params)
    
    def c3(self, data, **params):
        # min = -inf, max = inf
        return self._normalize(super().c3, data, **params)
    
    def time_reversal_asymmetry_statistic(self, data, **params):
        # min = -inf, max = inf
        return self._normalize(super().time_reversal_asymmetry_statistic, data, **params)
    
    ##########################################################################
    # 3. Complexity Features
    ##########################################################################
    
    def kullback_leibler_score(self, data, **params):
        # min = -inf, max = inf
        return self._normalize(super().kullback_leibler_score, data, **params)
    
    def cid_ce(self, data, **params):
        # min = 0, max = inf
        return self._normalize(super().cid_ce, data, minimum=0, **params)
