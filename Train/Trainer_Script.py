# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/31 10:53
import torch
from torch import nn
from Model import EEGNet, CCNN, SSVEPNet
from etc.global_config import config
from Utils import Constraint, LossFunction

def data_preprocess(EEGData_Train, EEGData_Test):
    '''
    Parameters
    ----------
    EEGData_Train: EEG Training Dataset (Including Data and Labels)
    EEGData_Test: EEG Testing Dataset (Including Data and Labels)

    Returns: Preprocessed EEG DataLoader
    -------
    '''
    algorithm = config['algorithm']
    ws = config["data_param"]["ws"]
    Fs = config["data_param"]["Fs"]
    bz = config[algorithm]["bz"]

    EEGData_Train, EEGLabel_Train = EEGData_Train[:]
    EEGData_Train = EEGData_Train[:, :, :, :int(Fs * ws)]
    if algorithm == "CCNN":
        EEGData_Train = CCNN.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
        EEGData_Train = torch.from_numpy(EEGData_Train)
    print("EEGData_Train.shape", EEGData_Train.shape)
    print("EEGLabel_Train.shape", EEGLabel_Train.shape)
    EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGLabel_Train)

    EEGData_Test, EEGLabel_Test = EEGData_Test[:]
    EEGData_Test = EEGData_Test[:, :, :, :int(Fs * ws)]
    if algorithm == "CCNN":
        EEGData_Test = CCNN.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
        EEGData_Test = torch.from_numpy(EEGData_Test)
    print("EEGData_Test.shape", EEGData_Test.shape)
    print("EEGLabel_Test.shape", EEGLabel_Test.shape)
    EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGLabel_Test)

    # Create DataLoader for the Dataset
    eeg_train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=bz, shuffle=True,
                                                   drop_last=True)
    eeg_test_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=bz, shuffle=False,
                                                   drop_last=True)

    return eeg_train_dataloader, eeg_test_dataloader

def build_model(devices):
    '''
    Parameters
    ----------
    device: the device to save DL models
    Returns: the building model
    -------
    '''
    algorithm = config['algorithm']
    Nc = config["data_param"]['Nc']
    Nf = config["data_param"]['Nf']
    Fs = config["data_param"]['Fs']
    ws = config["data_param"]['ws']
    lr = config[algorithm]['lr']
    wd = config[algorithm]['wd']

    if algorithm == "EEGNet":
        net = EEGNet.EEGNet(Nc, int(Fs * ws), Nf)

    elif algorithm == "CCNN":
        net = CCNN.CNN(Nc, 220, Nf)

    elif algorithm == "SSVEPNet":
        net = SSVEPNet.ESNet(Nc, int(Fs * ws), Nf)
        net = Constraint.Spectral_Normalization(net)

    net = net.to(devices)

    if algorithm == 'SSVEPNet':
        criterion = LossFunction.CELoss_Marginal_Smooth(Nf, stimulus_type='12')
    else:
        criterion = nn.CrossEntropyLoss(reduction="none")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)

    return net, criterion, optimizer