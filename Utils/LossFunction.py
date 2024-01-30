# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/3/29 19:17
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


        # print("att_lst:", self.attention_lst)

    def forward(self, outputs, targets):
        '''
        :param outputs: predictive results,shape:(batch_size, class_num)
        :param targets: ground truth,shape:(batch_size, )
        :return:
        '''
        targets_data = targets.cpu().data
        smoothed_labels = torch.empty(size=(outputs.shape[0], outputs.shape[1]))
        for i in range(smoothed_labels.shape[0]):
            label = targets_data[i]
            for j in range(smoothed_labels.shape[1]):
                if j == label:
                    smoothed_labels[i][j] = 1.0
                else:
                    smoothed_labels[i][j] = self.attention_lst[label][j]
        smoothed_labels = smoothed_labels.to(outputs.device)

        log_prob = F.log_softmax(outputs, dim=1)
        att_loss = - torch.sum(log_prob * smoothed_labels) / outputs.size(-2)
        ce_loss = nn.CrossEntropyLoss()(outputs, targets)
        loss_add = self.alpha * ce_loss + (1 - self.alpha) * att_loss
        return loss_add








