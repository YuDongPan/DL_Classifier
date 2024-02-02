# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/2/1 17:48
import torch
import torch.nn as nn
import torch.nn.functional as F


class convca(nn.Module):
    def __init__(self, Nc, Nt, Nf):
        super(convca, self).__init__()

        self.Nc = Nc
        self.Nt = Nt
        self.Nf = Nf

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(9, self.Nc),
                                 stride=(1, 1), padding="same")
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(1, self.Nc),
                                 stride=(1, 1), padding="same")
        self.conv1_3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, self.Nc),
                                 stride=(1, 1), padding="valid")
        self.dropout1 = nn.Dropout(0.75)

        self.conv2_1 = nn.Conv2d(in_channels=self.Nc, out_channels=40, kernel_size=(9, 1), padding="same")
        self.conv2_2 = nn.Conv2d(in_channels=40, out_channels=1, kernel_size=(9, 1), padding="same")
        self.dropout2 = nn.Dropout(0.15)

        self.dense = nn.Linear(self.Nf, self.Nf)

    def corr(self, input):
        x = input[0].squeeze(dim=1)  # (bz, 1, Nt, 1) => (bz, Nt, 1)
        t = input[1].squeeze(dim=1)  # (bz, 1, Nt, Nf) => (bz, Nt, Nf)
        t_ = t.view(-1, self.Nt, self.Nf)

        corr_xt = torch.einsum('ijk,ijl->ilk', [x, t_])  # (bz, Nf, 1)
        corr_xx = torch.einsum('ijk,ijk->ik', [x, x])    # (bz, 1)
        corr_tt = torch.einsum('ijl,ijl->il', [t_, t_])  # (bz, Nf)
        corr = torch.squeeze(corr_xt) / (torch.sqrt(corr_tt) * torch.sqrt(corr_xx))  # (bz, Nf)
        return corr

    def forward(self, sig, temp):
        '''
        Parameters
        ----------
        sig: bz × 1 × Nt × Nc
        temp: bz × Nc × Nt × Nf
        Returns: out (bz × Nf)
        -------
        '''
        sig = self.conv1_1(sig)
        sig = self.conv1_2(sig)
        sig = self.conv1_3(sig)
        sig = self.dropout1(sig)

        temp = self.conv2_1(temp)
        temp = self.conv2_2(temp)
        temp = self.dropout2(temp)

        corr = self.corr([sig, temp])

        out = self.dense(corr)
        return out
