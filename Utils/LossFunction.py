# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/29 19:17
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
class CELoss_Marginal_Smooth(nn.Module):

    def __init__(self, class_num, alpha=0.6, stimulus_type='12'):
        super(CELoss_Marginal_Smooth, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.stimulus_matrix_4 = [[0, 1],
                                  [2, 3]]

        self.stimulus_matrix_12 = [[0, 1, 2, 3],
                                   [4, 5, 6, 7],
                                   [8, 9, 10, 11]]

        if stimulus_type == '4':
            self.stimulus_matrix = self.stimulus_matrix_4

        elif stimulus_type == '12':
            self.stimulus_matrix = self.stimulus_matrix_12


        self.rows = len(self.stimulus_matrix[:])
        self.cols = len(self.stimulus_matrix[0])


        self.attention_lst = [[1.0 / (int(0 <= (i // self.cols - 1) <= self.rows - 1) +
                                      int(0 <= (i // self.cols + 1) <= self.rows - 1) +
                                      int(0 <= (i % self.cols - 1) <= self.cols - 1) +
                                      int(0 <= (i % self.cols + 1) <= self.cols - 1) +
                          int(0 <= (i // self.cols - 1) <= self.rows - 1 and 0 <= i % self.cols - 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols - 1) <= self.rows - 1 and 0 <= i % self.cols + 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols + 1) <= self.rows - 1 and 0 <= i % self.cols - 1 <= self.cols - 1) +
                          int(0 <= (i // self.cols + 1) <= self.rows - 1 and 0 <= i % self.cols + 1 <= self.cols - 1))
                               for j in range(class_num)] for i in range(class_num)]

        self.attention_lst = np.asarray(self.attention_lst)


    def forward(self, outputs, targets):
        '''
        :param outputs: predictive results, shape: (batch_size, class_num)
        :param targets: ground truth, shape: (batch_size,)
        :return:
        '''
        batch_size, class_num = outputs.shape

        # Obtain target labels and create smooth labels
        targets_data = targets.cpu().data
        smoothed_labels = torch.zeros(size=(batch_size, class_num), device=outputs.device)

        # Fill in the remaining parts of the attention matrix
        for i in range(smoothed_labels.shape[0]):
            label = targets_data[i]
            smoothed_labels[i] = torch.from_numpy(self.attention_lst[label])

        # Fill the smooth label with a position assignment of 1.0
        smoothed_labels[torch.arange(batch_size), targets_data] = 1.0

        # Calculate loss
        log_prob = F.log_softmax(outputs, dim=1)
        att_loss = -torch.sum(log_prob * smoothed_labels) / batch_size
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        loss_add = self.alpha * ce_loss + (1 - self.alpha) * att_loss

        return loss_add








