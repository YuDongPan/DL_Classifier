# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/30 17:16
import torch
import Utils.EEGDataset as EEGDataset
from Utils import Ploter
from Train import Classifier_Trainer, Trainer_Script
from etc.global_config import config

# 1、Define parameters of eeg
algorithm = config['algorithm']
'''Parameters for training procedure'''
Kf = config["train_param"]['Kf']
UD = config["train_param"]['UD']
ratio = config["train_param"]['ratio']

'''Parameters for ssvep data'''
ws = config["data_param"]["ws"]
Nh = config["data_param"]['Nh']
Nc = config["data_param"]['Nc']
Fs = config["data_param"]['Fs']
Nt = config["data_param"]['Nt']
Nf = config["data_param"]['Nf']
Ns = config["data_param"]['Ns']

'''Parameters for DL-based methods'''
epochs = config[algorithm]['epochs']
bz = config[algorithm]['bz']
lr = config[algorithm]['lr']
wd = config[algorithm]['wd']
lr_jitter = config[algorithm]['lr_jitter']

devices = "cuda" if torch.cuda.is_available() else "cpu"

# 2、Start Training
final_acc_list = []
for fold_num in range(Kf):
    final_test_acc_list = []
    print(f"Training for K_Fold {fold_num + 1}")
    for testSubject in range(1, Ns + 1):
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

        eeg_train_dataloader, eeg_test_dataloader = Trainer_Script.data_preprocess(EEGData_Train, EEGData_Test)

        # Define Network
        net, criterion, optimizer = Trainer_Script.build_model(devices)
        test_acc = Classifier_Trainer.train_on_batch(epochs, eeg_train_dataloader, eeg_test_dataloader, optimizer, criterion,
                                                      net, devices, lr_jitter=lr_jitter)
        final_test_acc_list.append(test_acc)

    final_acc_list.append(final_test_acc_list)


# 3、Plot Result
Ploter.plot_save_Result(final_acc_list, model_name=algorithm, dataset='DatasetA', UD=UD, ratio=ratio, win_size=str(ws),
                        text=True)