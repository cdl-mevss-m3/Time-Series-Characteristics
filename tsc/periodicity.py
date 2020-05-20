import numpy as np
from pandas import Timedelta, Series
from pandas import to_timedelta
from pandas.tseries.frequencies import to_offset
from scipy import signal


def _noise_limits(y):
    """
    Return upper and lower limits of a noise band. Values in this band can be considered as noise.
    :param y: The signal.
    :return: Tuple (lower, upper).
    """
    y_sorted = np.sort(y)
    s = np.vstack([np.arange(len(y_sorted)), y_sorted - y_sorted[0]]).T
    n = np.array([-s[0, 1] + s[-1, 1], -len(y) + 1])
    d = np.dot(s, n)
    i_max = np.argmax(d)
    i_min = np.argmin(d)
    y_upper = y_sorted[i_max]
    y_lower = y_sorted[i_min]
    return y_lower, y_upper


PERIODS = ["30d", "14d", "7d", "3d", "2d", "1d", "12h", "8h", "6h", "4h", "3h", "2h", "1h", "30min", "15min"]


def periodicity(data, periods: list, dt_min=None):
    """
    Return a pandas.Series with a periodicity score for a predefined set of potential periods (seasons) in the data.
    :param data: A pandas.Series with a DateTimeIndex index.
    :param periods: A list of time periods in string format (e.g.: ["2d", "12h", "30min"]).
    :param dt_min: The time interval between values of ``data`` in minutes. If None, ``data`` must have a
        DateTimeIndex with a set frequency (e.g., via ``data = data.asfreq("1min")``) so the time interval
        can be inferred (default: None = infer time interval from ``data``).
    :return: A pandas.Series with the periods as index and the score are the values.
    """
    t1 = data.index.min().ceil("1d")
    t2 = data.index.max().floor("1d")
    
    interval = (t2 - t1)
    
    # time interval in minutes
    if dt_min is None:
        dt_min = to_timedelta(to_offset(data.index.freq)).total_seconds() / 60
    
    result = []
    for p in periods:
        period = Timedelta(p)
        
        # there must be enough data for a period
        if interval < period * 4:
            result.append((period, np.nan))
            continue
        
        # base frequency of the period in 1/min
        f0 = 60 / period.total_seconds()
        
        # select an integer multiple of the period from the data
        t2 = t1 + (interval // period) * period
        selected = data[(data.index >= t1) & (data.index < t2)]
        
        # compute the periodogram
        fxx, pxx = signal.periodogram(selected.values, fs=dt_min, return_onesided=True, detrend=None, scaling="spectrum")
        
        # skip the offset
        fxx = fxx[1:]
        pxx = pxx[1:]
        
        # calculate upper limit of the periodogram
        pxx_lower, pxx_upper = _noise_limits(pxx)
        
        # set values below nose level to 0.0
        pxx[pxx < pxx_upper] = 0.0
        
        # normalize to norm 1.0
        pxx /= np.sqrt(np.sum(pxx ** 2))
        
        # mm, and mp are on off indices of the expected integer multiplies of the expected base frequency f0
        m1 = fxx / f0
        m1 = np.where(np.abs(m1 - np.round(m1)) / m1 < 1e-9)[0]
        
        mm = m1[m1 > 0]
        mm -= 1
        
        mp = m1[m1 < len(pxx) - 1]
        mp += 1
        
        # energy of the base frequency
        e0 = pxx[m1[0] - 1] + pxx[m1[0]] + pxx[m1[0] + 1]
        
        # all the energy above the base frequency f0
        norm = pxx[fxx >= f0].sum()
        
        # for a pronounced periodicity we expect high contributions from e0 as well as all harmonics
        # This is also just a heuristic and could be improved
        energy = e0 * (pxx[mm].sum() + pxx[m1].sum() + pxx[mp].sum())
        
        # normalize
        if norm > 0:
            energy /= norm
        
        result.append((period, energy))
    
    index, data = list(zip(*sorted(result)))
    pg = Series(index=index, data=data, name="periodicity")
    pg.index.name = "period"
    return pg
