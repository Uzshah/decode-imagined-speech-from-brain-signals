"""
Credit: (Time Series Feature Extraction Library) TSFEL by M. Barandas et al.
"""
import numpy as np
import scipy 
def compute_time(signal, fs):
    return np.arange(0, len(signal))/fs

def calc_ecdf(signal):
    """Computes the ECDF of the signal.
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
    """ This functions computes the fft of a signal.
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
#     fmag = fmag[~np.isnan(fmag)]
#     f = f[~np.isnan(f)]

    return f[:len(signal) // 2], fmag[:len(signal) // 2]



def spectral_spread(signal, fs):
    """
        Parameters
        ----------
        signal : nd-array
            Signal from which spectral spread is computed.
        fs : int
            Sampling frequency
        Returns
        -------
        float
            Spectral Spread"""
    
    f, fmag = calc_fft(signal, fs)
    f = f.tolist()
    fmag = fmag.tolist()
    spect_centroid = spectral_centroid(signal, fs)

    if not np.sum(fmag):
        return 0
    else:
        return np.dot(((f - spect_centroid) ** 2), (fmag / np.sum(fmag))) ** 0.5
    
def spectral_distance(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Signal from which spectral distance is computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        spectral distance
    """
    f, fmag = calc_fft(signal, fs)

    cum_fmag = np.cumsum(fmag)

    # Computing the linear regression
    points_y = np.linspace(0, cum_fmag[-1], len(cum_fmag))

    return np.sum(points_y - cum_fmag)


# @set_domain("domain", "spectral")
def fundamental_frequency(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Input from which fundamental frequency is computed
    fs : int
        Sampling frequency
    Returns
    -------
    f0: float
       Predominant frequency of the signal
    """
    signal = signal - np.mean(signal)
    f, fmag = calc_fft(signal, fs)

    # Finding big peaks, not considering noise peaks with low amplitude

    bp = scipy.signal.find_peaks(fmag, height=max(fmag) * 0.3)[0]

    # # Condition for offset removal, since the offset generates a peak at frequency zero
    bp = bp[bp != 0]
    if not list(bp):
        f0 = 0
    else:
        # f0 is the minimum big peak frequency
        f0 = f[min(bp)]

    return f0


# @set_domain("domain", "spectral")
def max_power_spectrum(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Input from which maximum power spectrum is computed
    fs : scalar
        Sampling frequency
    Returns
    -------
    nd-array
        Max value of the power spectrum density
    """
    if np.std(signal) == 0:
        return float(max(scipy.signal.welch(signal, fs, nperseg=len(signal))[1]))
    else:
        return float(max(scipy.signal.welch(signal / np.std(signal), fs, nperseg=len(signal))[1]))


# @set_domain("domain", "spectral")
def max_frequency(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Input from which maximum frequency is computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        0.95 of maximum frequency using cumsum
    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)

    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.95)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)

    return f[ind_mag]


# @set_domain("domain", "spectral")
def median_frequency(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Input from which median frequency is computed
    fs: int
        Sampling frequency
    Returns
    -------
    f_median : int
       0.50 of maximum frequency using cumsum.
    """
    f, fmag = calc_fft(signal, fs)
    cum_fmag = np.cumsum(fmag)
    try:
        ind_mag = np.where(cum_fmag > cum_fmag[-1] * 0.50)[0][0]
    except IndexError:
        ind_mag = np.argmax(cum_fmag)
    f_median = f[ind_mag]

    return f_median

def spectral_centroid(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Signal from which spectral centroid is computed
    fs: int
        Sampling frequency
    Returns
    -------
    float
        Centroid
    """
    f, fmag = calc_fft(signal, fs)
    if not np.sum(fmag):
        return 0
    else:
        return np.dot(f, fmag / np.sum(fmag))


# @set_domain("domain", "spectral")
def spectral_decrease(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Signal from which spectral decrease is computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        Spectral decrease
    """
    f, fmag = calc_fft(signal, fs)

    fmag_band = fmag[1:]
    len_fmag_band = np.arange(2, len(fmag) + 1)

    # Sum of numerator
    soma_num = np.sum((fmag_band - fmag[0]) / (len_fmag_band - 1), axis=0)

    if not np.sum(fmag_band):
        return 0
    else:
        # Sum of denominator
        soma_den = 1 / np.sum(fmag_band)

        # Spectral decrease computing
        return soma_den * soma_num


# @set_domain("domain", "spectral")
def spectral_kurtosis(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Signal from which spectral kurtosis is computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        Spectral Kurtosis
    """
    f, fmag = calc_fft(signal, fs)
    if not spectral_spread(signal, fs):
        return 0
    else:
        spect_kurt = ((f - spectral_centroid(signal, fs)) ** 4) * (fmag / np.sum(fmag))
        return np.sum(spect_kurt) / (spectral_spread(signal, fs) ** 4)


# @set_domain("domain", "spectral")
def spectral_skewness(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Signal from which spectral skewness is computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        Spectral Skewness
    """
    f, fmag = calc_fft(signal, fs)
    spect_centr = spectral_centroid(signal, fs)

    if not spectral_spread(signal, fs):
        return 0
    else:
        skew = ((f - spect_centr) ** 3) * (fmag / np.sum(fmag))
        return np.sum(skew) / (spectral_spread(signal, fs) ** 3)


# @set_domain("domain", "spectral")
def spectral_slope(signal, fs):
    """Misdariis P., McAdams S.
    Feature computational cost: 1
    Parameters
    ----------
    signal : nd-array
        Signal from which spectral slope is computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        Spectral Slope
    """
    f, fmag = calc_fft(signal, fs)
    sum_fmag = fmag.sum()
    dot_ff = (f * f).sum()
    sum_f = f.sum()
    len_f = len(f)

    if not ([f]) or (sum_fmag == 0):
        return 0
    else:
        if not (len_f * dot_ff - sum_f ** 2):
            return 0
        else:
            num_ = (1 / sum_fmag) * (len_f * np.sum(f * fmag) - sum_f * sum_fmag)
            denom_ = (len_f * dot_ff - sum_f ** 2)
            return num_ / denom_


# @set_domain("domain", "spectral")
def spectral_variation(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Signal from which spectral variation is computed.
    fs : int
        Sampling frequency
    Returns
    -------
    float
        Spectral Variation
    """
    f, fmag = calc_fft(signal, fs)

    sum1 = np.sum(np.array(fmag)[:-1] * np.array(fmag)[1:])
    sum2 = np.sum(np.array(fmag)[1:] ** 2)
    sum3 = np.sum(np.array(fmag)[:-1] ** 2)

    if not sum2 or not sum3:
        variation = 1
    else:
        variation = 1 - (sum1 / ((sum2 ** 0.5) * (sum3 ** 0.5)))

    return variation


# @set_domain("domain", "spectral")
def spectral_positive_turning(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Input from which the number of positive turning points of the fft magnitude are computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        Number of positive turning points
    """
    f, fmag = calc_fft(signal, fs)
    diff_sig = np.diff(fmag)

    array_signal = np.arange(len(diff_sig[:-1]))

    positive_turning_pts = np.where((diff_sig[array_signal + 1] < 0) & (diff_sig[array_signal] > 0))[0]

    return len(positive_turning_pts)




# @set_domain("domain", "spectral")
def spectral_roll_on(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Signal from which spectral roll-on is computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        Spectral roll-on
    """
    f, fmag = calc_fft(signal, fs)
    cum_ff = np.cumsum(fmag)
    value = 0.05 * (np.sum(fmag))

    return f[np.where(cum_ff >= value)[0][0]]

    
def power_bandwidth(signal, fs):
    """
    Parameters
    ----------
    signal : nd-array
        Input from which the power bandwidth computed
    fs : int
        Sampling frequency
    Returns
    -------
    float
        Occupied power in bandwidth
    """
    # Computing the power spectrum density
    if np.std(signal) == 0:
        freq, power = scipy.signal.welch(signal, fs, nperseg=len(signal))
    else:
        freq, power = scipy.signal.welch(signal / np.std(signal), fs, nperseg=len(signal))

    if np.sum(power) == 0:
        return 0.0

    # Computing the lower and upper limits of power bandwidth
    cum_power = np.cumsum(power)
    f_lower = freq[np.where(cum_power >= cum_power[-1] * 0.95)[0][0]]

    cum_power_inv = np.cumsum(power[::-1])
    f_upper = freq[np.abs(np.where(cum_power_inv >= cum_power[-1] * 0.95)[0][0] - len(power) + 1)]

    # Returning the bandwidth in terms of frequency

    return np.abs(f_upper - f_lower)


def fft_mean_coeff(signal, fs, nfreq=256):
    
    if nfreq > len(signal) // 2 + 1:
        nfreq = len(signal) // 2 + 1

    fmag_mean = scipy.signal.spectrogram(signal, fs, nperseg=nfreq * 2 - 2)[2].mean(1)

    return np.array(fmag_mean)


def spectral_entropy(signal, fs):
       
        # Removing DC component
    sig = signal - np.mean(signal)

    f, fmag = calc_fft(sig, fs)

    power = fmag ** 2

    if power.sum() == 0:
        return 0.0

    prob = np.divide(power, power.sum())

    prob = prob[prob != 0]

        # If probability all in one value, there is no entropy
    if prob.size == 1:
        return 0.0

    return -np.multiply(prob, np.log2(prob)).sum() / np.log2(prob.size)
