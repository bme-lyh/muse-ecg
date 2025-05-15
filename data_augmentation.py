import copy
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from tqdm import tqdm


def random_resize(signal, label, scale_range=(.5, 2)):
    def label_sharpen(label):
        label_sharpened = copy.deepcopy(label)
        label_sharpened[label>=.5] = 1
        label_sharpened[label<=.5] = 0
        return label_sharpened
    
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    length = signal.shape[0]
    new_length = int(length * scale_factor)
    signal_resized = zoom(signal, (new_length/length,1))
    label_resized = label_sharpen(zoom(label, (new_length/length,1)))

    if new_length<length:
        pad_length = length - new_length
        pad_l = np.random.randint(0, pad_length)
        pad_r = pad_length - pad_l
        signal_resized = np.pad(signal_resized, pad_width=((pad_l,pad_r),(0,0)), mode='constant', constant_values=0)
        label_resized = np.pad(label_resized, pad_width=((pad_l,pad_r),(0,0)), mode='constant', constant_values=0)
        label_resized[0:pad_l,0] = 1
        label_resized[-pad_r:,0] = 1
    else:
        signal_resized = signal_resized[0:length,:]
        label_resized = label_resized[0:length,:]

    return signal_resized, label_resized


def signal_power(signal):
    return np.mean((signal - np.median(signal))**2)


def additive_white_gaussian_noise(signal, snr=10):
    # Compute signal power
    power = signal_power(signal)
    # Calculate signal to noise ratio with randomness
    snr = snr + np.random.uniform(low=-snr/5, high=snr/5)
    power_noise = power/10**(snr/10.)
    noise = np.random.normal(0,np.sqrt(power_noise),len(signal))
    return noise


def baseline_wander_noise(signal, fs=500, snr=-10, freq=.15):
    # Compute signal power
    power = signal_power(signal)
    # Calculate signal to noise ratio with randomness
    snr = snr + np.random.uniform(low=-snr/5, high=snr/5)
    freq = freq + np.random.uniform(low=-freq/5, high=freq/5)
    NormFreq = 2.*np.pi*freq/fs
    power_noise = power/10**(snr/10.)
    Amplitude = np.sqrt(2*power_noise)
    noise = Amplitude*np.sin(NormFreq*np.arange(len(signal)) + np.random.uniform(low=-np.pi, high=np.pi)) # Random initial phase
    return noise


def zscore_normalize(signal, axis=-1):
    """
    Applies Z-score normalization to a signal (e.g., ECG).

    Args:
        signal: A NumPy array representing the signal. It can be 1D, 2D, or more.
        axis: The axis along which to calculate the mean and standard deviation.
            Typically, for ECG signals, this would be the time axis.

    Returns:
        A NumPy array of the same shape as the input signal, but Z-score normalized.
    """
    mean = np.nanmean(signal, axis=axis, keepdims=True)
    std = np.nanstd(signal, axis=axis, keepdims=True)

    # Handle cases where the standard deviation is zero (to avoid division by zero)
    if np.all(std == 0):
        raise ValueError("The standard deviation of the signal is zero. Cannot normalize the signal.")
    else:
        signal_normalized = (signal - mean) / std
        signal_normalized = np.nan_to_num(signal_normalized)
        return signal_normalized
    

def rand_mask(signal, label=None, mask_ratio=0.1):
    length = signal.shape[0]
    preserved_ratio = 1 - mask_ratio
    preserved_length = int(length * preserved_ratio)
    start = np.random.randint(0, length - preserved_length)
    signal[0:start] = np.nan
    signal[start+preserved_length:] = np.nan
    if label is None:
        return signal
    else:
        label[start:start+preserved_length, 0] = 1
        label[start:start+preserved_length, 1:] = 0
        return signal, label