import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AMIML(nn.Module):

    def __init__(self, input_dim=256, R=100, alpha=0.1,d=8):
        super(HE2RNA, self).__init__()

        self.input_dim = input_dim
        self.minmax = R
        self.alpha = alpha
        self.d = d

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)

        self.conv1 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0, dilation=1,
                               bias=True)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1,
                               bias=True)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, dilation=1,
                               bias=True)
        self.bn3 = nn.BatchNorm1d(32)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=self.d, kernel_size=1, stride=1, padding=0, dilation=1,
                               bias=True)
        self.bn4 = nn.BatchNorm1d(self.d)

        self.conv5 = nn.Conv1d(in_channels=self.d, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1,
                               bias=True)

        self.conv6 = nn.ConvTranspose1d(in_channels=1, out_channels=self.d, kernel_size=1, stride=1, padding=0, dilation=1,
                                        bias=True)
        self.bn5 = nn.BatchNorm1d(self.d)

        self.query = nn.Linear(self.d, self.d)
        self.key = nn.Linear(self.d, self.d)
        self.value = nn.Linear(self.d, self.d)

        self.mp = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x4 = self.conv4(x)

        x = self.conv5(x4)

        sorted1, indices1 = torch.topk(x, self.minmax, dim=2, largest=True, sorted=True)
        sorted2, indices2 = torch.topk(x, self.minmax, dim=2, largest=False, sorted=True)
        #
        indices1 = torch.cat((indices1, indices1), dim=1)
        indices2 = torch.cat((indices2, indices2), dim=1)
        indices1 = torch.cat((indices1, indices1), dim=1)
        indices2 = torch.cat((indices2, indices2), dim=1)
        indices1 = torch.cat((indices1, indices1), dim=1)
        indices2 = torch.cat((indices2, indices2), dim=1)

        x_sort1 = torch.gather(x4, 2, indices1)
        x_sort2 = torch.gather(x4, 2, indices2)

        x = torch.cat((sorted1, sorted2), dim=2)
        x_cat = torch.cat((x_sort1, x_sort2), dim=2)
        res = self.conv6(x)

        x = self.alpha*res + x_cat

        x = x.permute(0, 2, 1)

        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(x)
        mixed_value_layer = self.value(x)

        xx = torch.matmul(mixed_query_layer, mixed_key_layer.permute(0, 2, 1))
        xx = self.dropout(xx)

        attention_scores = xx / math.sqrt(self.d)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        x = torch.matmul(attention_probs, mixed_value_layer)
        x = self.dropout(x)

        x = torch.mean(x, dim=1, keepdim=True)
        x = x.squeeze(axis=1)

        y = F.softmax(x, dim=1)

        return y


if __name__ == '__main__':
    net = AMIML(input_dim=256, R=100,alpha=0.1,d=8)
    print(net)