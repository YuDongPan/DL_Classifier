# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/31 10:53
import numpy as np
import torch
from torch import nn
from Model import EEGNet, CCNN, SSVEPNet, FBtCNN, ConvCA, SSVEPformer, DDGCNN
from Utils import Constraint, LossFunction, Script
from etc.global_config import config

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
    Nf = config["data_param"]["Nf"]
    bz = config[algorithm]["bz"]


    '''Loading Training Data'''
    EEGData_Train, EEGLabel_Train = EEGData_Train[:]
    EEGData_Train = EEGData_Train[:, :, :, :int(Fs * ws)]

    if algorithm == "ConvCA":
        EEGData_Train = torch.swapaxes(EEGData_Train, axis0=2, axis1=3) # (Nh, 1, Nt, Nc)
        EEGTemp_Train = Script.get_Template_Signal(EEGData_Train, Nf)  # (Nf × 1 × Nt × Nc)
        EEGTemp_Train = torch.swapaxes(EEGTemp_Train, axis0=0, axis1=1)  # (1 × Nf × Nt × Nc)
        EEGTemp_Train = EEGTemp_Train.repeat((EEGData_Train.shape[0], 1, 1, 1))  # (Nh × Nf × Nt × Nc)
        EEGTemp_Train = torch.swapaxes(EEGTemp_Train, axis0=1, axis1=3)  # (Nh × Nc × Nt × Nf)

        print("EEGData_Train.shape", EEGData_Train.shape)
        print("EEGTemp_Train.shape", EEGTemp_Train.shape)
        print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGTemp_Train, EEGLabel_Train)

    else:
        if algorithm == "CCNN":
            EEGData_Train = CCNN.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)

        elif algorithm == "SSVEPformer":
            EEGData_Train = SSVEPformer.complex_spectrum_features(EEGData_Train.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Train = torch.from_numpy(EEGData_Train)
            EEGData_Train = EEGData_Train.squeeze(1)

        elif algorithm == "DDGCNN":
            EEGData_Train = torch.swapaxes(EEGData_Train, axis0=1, axis1=3)  # (Nh, 1, Nc, Nt) => (Nh, Nt, Nc, 1)

        print("EEGData_Train.shape", EEGData_Train.shape)
        print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGLabel_Train)


    '''Loading Testing Data'''
    EEGData_Test, EEGLabel_Test = EEGData_Test[:]
    EEGData_Test = EEGData_Test[:, :, :, :int(Fs * ws)]

    if algorithm == "ConvCA":
        EEGData_Test = torch.swapaxes(EEGData_Test, axis0=2, axis1=3)  # (Nh, 1, Nt, Nc)
        EEGTemp_Test = Script.get_Template_Signal(EEGData_Test, Nf)  # (Nf × 1 × Nt × Nc)
        EEGTemp_Test = torch.swapaxes(EEGTemp_Test, axis0=0, axis1=1)  # (1 × Nf × Nt × Nc)
        EEGTemp_Test = EEGTemp_Test.repeat((EEGData_Test.shape[0], 1, 1, 1))  # (Nh × Nf × Nt × Nc)
        EEGTemp_Test = torch.swapaxes(EEGTemp_Test, axis0=1, axis1=3)  # (Nh × Nc × Nt × Nf)

        print("EEGData_Test.shape", EEGData_Test.shape)
        print("EEGTemp_Test.shape", EEGTemp_Test.shape)
        print("EEGLabel_Test.shape", EEGLabel_Test.shape)
        EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGTemp_Test, EEGLabel_Test)

    else:
        if algorithm == "CCNN":
            EEGData_Test = CCNN.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)

        elif algorithm == "SSVEPformer":
            EEGData_Test = SSVEPformer.complex_spectrum_features(EEGData_Test.numpy(), FFT_PARAMS=[Fs, ws])
            EEGData_Test = torch.from_numpy(EEGData_Test)
            EEGData_Test = EEGData_Test.squeeze(1)

        elif algorithm == "DDGCNN":
            EEGData_Test = torch.swapaxes(EEGData_Test, axis0=1, axis1=3)  # (Nh, 1, Nc, Nt) => (Nh, Nt, Nc, 1)

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
    Nt = int(Fs * ws)

    if algorithm == "EEGNet":
        net = EEGNet.EEGNet(Nc, Nt, Nf)

    elif algorithm == "CCNN":
        net = CCNN.CNN(Nc, 220, Nf)

    elif algorithm == "FBtCNN":
        net = FBtCNN.tCNN(Nc, Nt, Nf, Fs)

    elif algorithm == "ConvCA":
        net = ConvCA.convca(Nc, Nt, Nf)

    elif algorithm == "SSVEPformer":
        net = SSVEPformer.SSVEPformer(depth=2, attention_kernal_length=31, chs_num=Nc, class_num=Nf,
                                      dropout=0.5)
        net.apply(Constraint.initialize_weights)

    elif algorithm == "SSVEPNet":
        net = SSVEPNet.ESNet(Nc, Nt, Nf)
        net = Constraint.Spectral_Normalization(net)

    elif algorithm == "DDGCNN":
        bz = config[algorithm]["bz"]
        norm = config[algorithm]["norm"]
        act = config[algorithm]["act"]
        trans_class = config[algorithm]["trans_class"]
        n_filters = config[algorithm]["n_filters"]
        net = DDGCNN.DenseDDGCNN([bz, Nt, Nc], k_adj=3, num_out=n_filters, dropout=0.5, n_blocks=3, nclass=Nf,
                                 bias=False, norm=norm, act=act, trans_class=trans_class, device=devices)

    net = net.to(devices)

    if algorithm == 'SSVEPNet':
        criterion = LossFunction.CELoss_Marginal_Smooth(Nf, stimulus_type='12')
    else:
        criterion = nn.CrossEntropyLoss(reduction="none")

    if algorithm == "SSVEPformer":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=wd)

    return net, criterion, optimizer