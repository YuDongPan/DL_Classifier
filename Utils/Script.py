# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/2/2 16:57
import torch
import numpy as np
from scipy import signal


def cal_Parameters(net, name):
    '''
    :param net: input neural network
    :return: number of parameters of network
    '''
    total = sum([param.nelement() for param in net.parameters()])
    print(f"Number of parameters of {name}: {total / 1e6:.2f}M")


def norm_Data(X):
    '''
    :param X: input EEG Data
    :return: Normalizaed EEG
    '''
    X = X.numpy()
    Nh = X.shape[0]
    Nc = X.shape[2]
    for h in range(Nh):
        for c in range(Nc):
            X[h, 0, c, :] = (X[h, 0, c, :] - np.mean(X[h, 0, c, :])) / np.std(X[h, 0, c, :])

    norm_X = torch.from_numpy(X)
    return norm_X


def filter_Data(X, Fs, low, high, N=4, type="bandpass"):
    '''
    :param X: the eeg data to be filtered
    :param Fs: sample rate
    :param low: lowest bound frequency
    :param high: highest bound frequency
    :param N: filter-order
    :return: filtered eeg data
    '''
    b, a = 0, 0
    if type == "bandpass":
        b, a = signal.butter(N, [2 * low / Fs, 2 * high / Fs], type)

    elif type == "highpass":
        b, a = signal.butter(N, 2 * low / Fs, type)

    elif type == "lowpass":
        b, a = signal.butter(N, 2 * high / Fs, type)

    X = signal.filtfilt(b, a, X, axis=-1)

    return X


def get_Template_Signal(X, Nf):
    '''
    :param X: input eeg data (Nh × 1 × Nc × Nt)
    :param Nf: number of flicker stimulus
    :return: template eeg signal (Nf × 1 × Nc × Nt)
    '''
    X = X.numpy()
    reference_signals = []
    num_per_cls = X.shape[0] // Nf
    for cls_num in range(Nf):
        reference_f = X[cls_num * num_per_cls:(cls_num + 1) * num_per_cls]
        reference_f = np.mean(reference_f, axis=0)
        reference_signals.append(reference_f)
    reference_signals = np.asarray(reference_signals)
    reference_signals = torch.from_numpy(reference_signals)
    return reference_signals


def EEG_Data_Segment(eeg_data, label_data, Ns):
    '''
    :param eeg_data: input eeg data
    :param label_data: input label data
    :param Ns: number of segmented sample points
    :return: segmented eeg data
    '''
    Nh = eeg_data.shape[0]
    Nc = eeg_data.shape[2]
    Nt = eeg_data.shape[3]
    num_segments = Nt // Ns
    segment_eeg_data = torch.zeros((Nh * num_segments, 1, Nc, Ns))
    segment_label_data = torch.zeros((Nh * num_segments, 1))
    for h in range(Nh):
        for s in range(num_segments):
            segment_eeg_data[h * num_segments + s, :, :, :] = eeg_data[h, :, :, s * Ns: (s + 1) * Ns]
            segment_label_data[h * num_segments + s, :] = label_data[h]

    print("segment success!")
    return segment_eeg_data, segment_label_data
