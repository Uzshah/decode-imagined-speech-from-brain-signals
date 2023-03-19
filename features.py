"""
Credit: (Time Series Feature Extraction Library) TSFEL by M. Barandas et al.
"""
import numpy as np
import scipy 
def compute_time(signal, fs):
    return np.arange(0, len(signal))/fs

def calc_ecdf(signal):
    """
      Parameters
      ----------
      signal : nd-array
          Input from which ECDF is computed
      Returns
      -------
      nd-array
        Sorted signal and computed ECDF.
      """
    return np.sort(signal), np.arange(1, len(signal)+1)/len(signal)


def calc_fft(signal, fs):
    """ 
    Parameters
    ----------
    signal : nd-array
        The input signal from which fft is computed
    fs : int
        Sampling frequency
    Returns
    -------
    f: nd-array
        Frequency values (xx axis)
    fmag: nd-array
        Amplitude of the frequency values (yy axis)
    """

    fmag = np.abs(np.fft.fft(signal))
    f = np.linspace(0, fs // 2, len(signal) // 2)

    return f[:len(signal) // 2].copy(), fmag[:len(signal) // 2].copy()


class features(object):
    '''Temporal features'''
    def calc_centroid(signal, fs):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which centroid is computed
        fs: int
            Signal sampling frequency

        Returns
        -------
        float
            Temporal centroid

        """

        time = compute_time(signal, fs)

        energy = np.array(signal) ** 2

        t_energy = np.dot(np.array(time), np.array(energy))
        energy_sum = np.sum(energy)

        if energy_sum == 0 or t_energy == 0:
            centroid = 0
        else:
            centroid = t_energy / energy_sum

        return centroid
    
    def mean_abs_diff(signal):
        """
       Parameters
       ----------
       signal : nd-array
           Input from which mean absolute deviation is computed

       Returns
       -------
       float
           Mean absolute difference result

       """
        return np.mean(np.abs(np.diff(signal)))
    
    def mean_diff(signal):
        """
       Parameters
       ----------
       signal : nd-array
           Input from which mean of differences is computed

       Returns
       -------
       float
           Mean difference result

       """
        return np.mean(np.diff(signal))
    
    def median_abs_diff(signal):
        """
       Parameters
       ----------
       signal : nd-array
           Input from which median absolute difference is computed

       Returns
       -------
       float
           Median absolute difference result

       """
        return np.median(np.abs(np.diff(signal)))
    
    def median_diff(signal):
        """
       Parameters
       ----------
       signal : nd-array
           Input from which median of differences is computed

       Returns
       -------
       float
           Median difference result

       """
        return np.median(np.diff(signal))
    
    def distance(signal):
        """
        Parameters
        ----------
        signal : nd-array
            Input from which distance is computed

        Returns
        -------
        float
            Signal distance

        """
        diff_sig = np.diff(signal).astype(float)
        return np.sum([np.sqrt(1 + diff_sig ** 2)])
    
    def sum_abs_diff(signal):
        """
       Parameters
       ----------
       signal : nd-array
           Input from which sum absolute difference is computed

       Returns
       -------
       float
           Sum absolute difference result

       """
        return np.sum(np.abs(np.diff(signal)))
    
    def slope(signal):
        """
        Parameters
        ----------
        signal : nd-array
            Input from which linear equation is computed

        Returns
        -------
        float
            Slope

        """
        t = np.linspace(0, len(signal) - 1, len(signal))

        return np.polyfit(t, signal, 1)[0]
    
    def auc(signal, fs):
        """
        Parameters
        ----------
        signal : nd-array
            Input from which the area under the curve is computed
        fs : int
            Sampling Frequency
        Returns
        -------
        float
            The area under the curve value

    """
        t = compute_time(signal, fs)

        return np.sum(0.5 * np.diff(t) * np.abs(np.array(signal[:-1]) + np.array(signal[1:])))
    
    def pk_pk_distance(signal):
        """
        Parameters
        ----------
        signal : nd-array
            Input from which the area under the curve is computed

        Returns
        -------
        float
            peak to peak distance

        """
        return np.abs(np.max(signal) - np.min(signal))
    
    def entropy(signal, prob='standard'):
        """
        Parameters
        ----------
        signal : nd-array
            Input from which entropy is computed
        prob : string
            Probability function (kde or gaussian functions are available)

        Returns
        -------
        float
            The normalized entropy value

        """

        if prob == 'standard':
            value, counts = np.unique(signal, return_counts=True)
            p = counts / counts.sum()
        elif prob == 'kde':
            p = kde(signal)
        elif prob == 'gauss':
            p = gaussian(signal)

        if np.sum(p) == 0:
            return 0.0

        # Handling zero probability values
        p = p[np.where(p != 0)]

        # If probability all in one value, there is no entropy
        if np.log2(len(signal)) == 1:
            return 0.0
        elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
            return 0.0
        else:
            return - np.sum(p * np.log2(p)) / np.log2(len(signal))
    
    def neighbourhood_peaks(signal, n=10):
        """
        Parameters
        ----------
        signal : nd-array
             Input from which the number of neighbourhood peaks is computed
        n :  int
            Number of peak's neighbours to the left and to the right

        Returns
        -------
        int
            The number of peaks from a defined neighbourhood of the signal
        """
        signal = np.array(signal)
        subsequence = signal[n:-n]
        # initial iteration
        peaks = ((subsequence > np.roll(signal, 1)[n:-n]) & (subsequence > np.roll(signal, -1)[n:-n]))
        for i in range(2, n + 1):
            peaks &= (subsequence > np.roll(signal, i)[n:-n])
            peaks &= (subsequence > np.roll(signal, -i)[n:-n])
        return np.sum(peaks)
    
    

###################################################Statictical features ##################################################################



    def hist(signal, nbins=10, r=1):
        """

        Parameters
        ----------
        signal : nd-array
            Input from histogram is computed
        nbins : int
            The number of equal-width bins in the given range
        r : float
            The lower(-r) and upper(r) range of the bins

        Returns
        -------
        nd-array
            The values of the histogram

        """
        histsig, bin_edges = np.histogram(signal, bins=nbins, range=[-r, r])  # TODO:subsampling parameter

        return list(histsig)
    
    def interq_range(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which interquartile range is computed

        Returns
        -------
        float
            Interquartile range result

        """
        return np.percentile(signal, 75) - np.percentile(signal, 25)
    
    def kurtosis(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which kurtosis is computed

        Returns
        -------
        float
            Kurtosis result

        """
        return scipy.stats.kurtosis(signal)
    
    def skewness(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which skewness is computed

        Returns
        -------
        int
            Skewness result

        """
        return scipy.stats.skew(signal)
    
    def calc_max(signal):
        """

        Parameters
        ----------
        signal : nd-array
           Input from which max is computed

        Returns
        -------
        float
            Maximum result

        """
        return np.max(signal)
    
    def calc_min(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which min is computed

        Returns
        -------
        float
            Minimum result

        """
        return np.min(signal)
    
    def calc_mean(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which mean is computed.

        Returns
        -------
        float
            Mean result

        """
        return np.mean(signal)
    
    def calc_median(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which median is computed

        Returns
        -------
        float
            Median result

        """
        return np.median(signal)
    
    
    def mean_abs_deviation(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which mean absolute deviation is computed

        Returns
        -------
        float
            Mean absolute deviation result

        """
        return np.mean(np.abs(signal - np.mean(signal, axis=0)), axis=0)
    
    def median_abs_deviation(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which median absolute deviation is computed

        Returns
        -------
        float
            Mean absolute deviation result

        """
        return scipy.stats.median_absolute_deviation(signal, scale=1)
    
    def calc_std(signal):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which std is computed

        Returns
        -------
        float
            Standard deviation result

        """
        return np.std(signal)

    def calc_var(signal):
        """

        Parameters
        ----------
        signal : nd-array
           Input from which var is computed

        Returns
        -------
        float
            Variance result

        """
        return np.var(signal)
    
    def ecdf(signal, d=10):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which ECDF is computed
        d: integer
            Number of ECDF values to return

        Returns
        -------
        float
            The values of the ECDF along the time axis
        """
        _, y = calc_ecdf(signal)
        if len(signal) <= d:
            return list(y)
        else:
            return list(y[:d])
        
    def ecdf_percentile(signal, percentile=[0.2, 0.8]):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which ECDF is computed
        percentile: list
            Percentile value to be computed

        Returns
        -------
        float
            The input value(s) of the ECDF
        """
        signal = np.array(signal)
        if isinstance(percentile, str):
            percentile = eval(percentile)
        if isinstance(percentile, (float, int)):
            percentile = [percentile]

        # calculate ecdf
        x, y = calc_ecdf(signal)

        if len(percentile) > 1:
            # check if signal is constant
            if np.sum(np.diff(signal)) == 0:
                return list(np.repeat(signal[0], len(percentile)))
            else:
                return list([x[y <= p].max() for p in percentile])
        else:
            # check if signal is constant
            if np.sum(np.diff(signal)) == 0:
                return signal[0]
            else:
                return x[y <= percentile].max()
            
    def ecdf_percentile_count(signal, percentile=[0.2, 0.8]):
        """

        Parameters
        ----------
        signal : nd-array
            Input from which ECDF is computed
        percentile: list
            Percentile threshold

        Returns
        -------
        float
            The cumulative sum of samples
        """
        signal = np.array(signal)
        if isinstance(percentile, str):
            percentile = eval(percentile)
        if isinstance(percentile, (float, int)):
            percentile = [percentile]

        # calculate ecdf
        x, y = calc_ecdf(signal)

        if len(percentile) > 1:
            # check if signal is constant
            if np.sum(np.diff(signal)) == 0:
                return list(np.repeat(signal[0], len(percentile)))
            else:
                return list([x[y <= p].shape[0] for p in percentile])
        else:
            # check if signal is constant
            if np.sum(np.diff(signal)) == 0:
                return signal[0]
            else:
                return x[y <= percentile].shape[0]
         