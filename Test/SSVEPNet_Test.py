# Designer:Pan YuDong
# Coder:God's hand
# Time:2021/11/26 19:52
import torch
import argparse
import Utils.EEGDataset as EEGDataset
from Utils import Ploter
from Utils import LossFunction
from Model import SSVEPNet
from Utils import Constraint
from Train import Classifier_Trainer

# 1、Define parameters of eeg
'''                    
---------------------------------------------Intra-subject Experiments ---------------------------------
                        epochs    bz     lr   lr_scheduler    ws      Fs    Nt   Nc   Nh   Ns     wd
    DatasetA(1S/0.5S):  500      30    0.01      Y           1/0.5   256   1024  8   180  10  0.0003
    DatasetB(1S/0.5S):  500      16    0.01      Y           1/0.5   250   1000  8    80  10  0.0003
---------------------------------------------Inter-subject Experiments ---------------------------------
                        epochs     bz          lr       lr_scheduler  ws      Fs     Nt    Nc   Nh      wd        Kf
    DatasetA(1S/0.5S):  500/100   64/30    0.001/0.01     N/Y        1/0.5    256   1024   8    180   0/0.0001   1/5
    DatasetB(1S/0.5S):  500/100   64/30    0.001/0.01     N/Y        1/0.5    250   1000   8     80   0/0.0003   1/5 
'''
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500, help="number of epochs")
parser.add_argument('--bz', type=int, default=30, help="number of batch")
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--ws', type=float, default=1.0, help="window size of ssvep")
parser.add_argument('--Kf', type=int, default=1, help="k-fold cross validation")
parser.add_argument('--Nh', type=int, default=180, help="number of trial")
parser.add_argument('--Nc', type=int, default=8, help="number of channel")
parser.add_argument('--Fs', type=int, default=256, help="frequency of sample")
parser.add_argument('--Nt', type=int, default=1024, help="number of sample")
parser.add_argument('--Nf', type=int, default=12, help="number of stimulus")
parser.add_argument('--Ns', type=int, default=10, help="number of subjects")
parser.add_argument('--wd', type=int, default=0.0003, help="weight decay")
opt = parser.parse_args()
devices = "cuda" if torch.cuda.is_available() else "cpu"

# 2、Start Training
best_acc_list = []
final_acc_list = []
for fold_num in range(opt.Kf):
    best_valid_acc_list = []
    final_valid_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for testSubject in range(1, opt.Ns + 1):
        # **************************************** #
        '''12-class SSVEP Dataset'''
        # -----------Intra-Subject Experiments--------------
        # EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=opt.Kf,
        #                                           mode='test')
        # EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, KFold=fold_num, n_splits=opt.Kf,
        #                                          mode='train')

        EEGData_Train = EEGDataset.getSSVEP12Intra(subject=testSubject, train_ratio=0.2, mode='train')
        EEGData_Test = EEGDataset.getSSVEP12Intra(subject=testSubject, train_ratio=0.2, mode='test')

        # -----------Inter-Subject Experiments--------------
        # EEGData_Train = EEGDataset.getSSVEP12Inter(subject=testSubject, mode='train')
        # EEGData_Test = EEGDataset.getSSVEP12Inter(subject=testSubject, mode='test')

        EEGData_Train, EEGLabel_Train = EEGData_Train[:]
        EEGData_Train = EEGData_Train[:, :, :, :int(opt.Fs * opt.ws)]
        print("EEGData_Train.shape", EEGData_Train.shape)
        print("EEGLabel_Train.shape", EEGLabel_Train.shape)
        EEGData_Train = torch.utils.data.TensorDataset(EEGData_Train, EEGLabel_Train)

        EEGData_Test, EEGLabel_Test = EEGData_Test[:]
        EEGData_Test = EEGData_Test[:, :, :, :int(opt.Fs * opt.ws)]
        print("EEGData_Test.shape", EEGData_Test.shape)
        print("EEGLabel_Test.shape", EEGLabel_Test.shape)
        EEGData_Test = torch.utils.data.TensorDataset(EEGData_Test, EEGLabel_Test)

        # Create DataLoader for the Dataset
        train_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Train, batch_size=opt.bz, shuffle=True,
                                           drop_last=True)
        valid_dataloader = torch.utils.data.DataLoader(dataset=EEGData_Test, batch_size=opt.bz, shuffle=False,
                                           drop_last=True)

        # Define Network
        net = SSVEPNet.ESNet(opt.Nc, int(opt.Fs * opt.ws), opt.Nf)
        net = Constraint.Spectral_Normalization(net)
        net = net.to(devices)
        # criterion = nn.CrossEntropyLoss(reduction="none")
        criterion = LossFunction.CELoss_Marginal_Smooth(opt.Nf, stimulus_type='12')
        valid_acc = Classifier_Trainer.train_on_batch(opt.epochs, train_dataloader, valid_dataloader, opt.lr, criterion,
                                                      net, devices, wd=opt.wd, lr_jitter=True)
        final_valid_acc_list.append(valid_acc)

    final_acc_list.append(final_valid_acc_list)


# 3、Plot Result
Ploter.plot_save_Result(final_acc_list, model_name='SSVEPNet', dataset='DatasetA', UD=0, ratio=3, win_size=str(opt.ws),
                        text=True)
