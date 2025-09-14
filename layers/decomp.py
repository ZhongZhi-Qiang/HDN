import torch
from torch import nn
import torch.nn.functional as F
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [batch, seq_len, channels], e.g., [32, 114, 8]
        # 增加填充量以补偿 nn.AvgPool1d 的长度减少
        padding_left = (self.kernel_size - 1) // 2
        padding_right = self.kernel_size - 1 - padding_left  # 确保总填充量为 kernel_size - 1
        front = x[:, 0:1, :].repeat(1, padding_left, 1)
        end = x[:, -1:, :].repeat(1, padding_right, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size, configs=None):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
        