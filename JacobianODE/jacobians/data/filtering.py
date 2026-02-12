"""Signal filtering utilities for time series data."""

import numpy as np
import scipy.signal as signal


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='high', analog=False, output='sos')
    return sos

def butter_highpass_filter(data, cutoff, fs, order=2, bidirectional=True):
    sos = butter_highpass(cutoff, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    return sos

def butter_lowpass_filter(data, cutoff, fs, order=2, bidirectional=True):
    sos = butter_lowpass(cutoff, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

# Define the bandstop filter function
def butter_bandstop_filter(data, lowcut, highcut, fs, order=2, bidirectional=True):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, bidirectional=True):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    if bidirectional:
        y = signal.sosfiltfilt(sos, data)
    else:
        y = signal.sosfilt(sos, data)
    return y

def filter_data(data, low_pass=None, high_pass=None, dt=0.001, order=2, bidirectional=True):
    """
    Apply Butterworth filtering to time series data.

    This function can apply low-pass, high-pass, band-pass, or band-stop filtering
    to the input data using Butterworth filters.

    Parameters
    ----------
    data : np.ndarray
        Input time series data of shape (time_steps, n_dims)
    low_pass : float, optional
        Low-pass cutoff frequency, by default None
    high_pass : float, optional
        High-pass cutoff frequency, by default None
    dt : float, optional
        Time step size, by default 0.001
    order : int, optional
        Filter order, by default 2
    bidirectional : bool, optional
        Whether to apply the filter bidirectionally to avoid phase shifts, by default True

    Returns
    -------
    np.ndarray
        Filtered data of the same shape as the input

    Notes
    -----
    - If both low_pass and high_pass are None, returns original data
    - If only one cutoff is provided, applies either low-pass or high-pass filter
    - If both cutoffs are provided:
        - If low_pass > high_pass: applies band-pass filter
        - If low_pass < high_pass: applies band-stop filter
        - If low_pass == high_pass: returns original data
    """
    if low_pass is None and high_pass is None:
        return data
    elif low_pass is None and high_pass is not None:
        data_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):
            data_filt[:, i] = butter_highpass_filter(data[:, i], high_pass, 1/dt, order=order, bidirectional=bidirectional)
        return data_filt
    elif low_pass is not None and high_pass is None:
        data_filt = np.zeros(data.shape)
        for i in range(data.shape[1]):
            data_filt[:, i] = butter_lowpass_filter(data[:, i], low_pass, 1/dt, order=order, bidirectional=bidirectional)
        return data_filt
    else:
        if low_pass == high_pass:
            return data
        elif low_pass > high_pass:
            data_filt = np.zeros(data.shape)
            for i in range(data.shape[1]):
                data_filt[:, i] = butter_bandpass_filter(data[:, i], high_pass, low_pass, 1/dt, order=order, bidirectional=bidirectional)
            return data_filt
        else: # low_pass < high_pass
            data_filt = np.zeros(data.shape)
            for i in range(data.shape[1]):
                data_filt[:, i] = butter_bandstop_filter(data[:, i], low_pass, high_pass, 1/dt, order=order, bidirectional=bidirectional)
            return data_filt
