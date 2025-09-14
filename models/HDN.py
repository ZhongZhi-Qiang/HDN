import torch
import torch.nn as nn
import copy
from torch.nn.init import xavier_normal_, constant_
import torch.nn.functional as F
from layers.decomp import series_decomp

"""
Using per instance per variate period weight
"""

class Statistics_prediction(nn.Module):
    def __init__(self, configs, device):
        super(Statistics_prediction, self).__init__()
        self.device = device
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_list = configs.period_list
        self.k = configs.top_k
        self.channels = configs.enc_in if configs.features == 'M' else 1
        self.station_type = configs.station_type
        self.epsilon = 1e-5
        self._build_model()
        # self.decomp = series_decomp(configs.kernel_size)
        
    def _build_model(self):
        self.stat_predict = []
        self.decomps = []
        self.k = self.configs.top_k
        for i in range(self.k):
            period_len = self.configs.period_list[i]
            if self.configs.seq_len % period_len != 0:
                seq_len = (self.configs.seq_len // period_len) + 1
            else:
                seq_len = self.configs.seq_len // period_len
            
            if self.configs.pred_len % period_len != 0:
                pred_len = (self.configs.pred_len // period_len) + 1
            else:
                pred_len = self.configs.pred_len // period_len

            self.stat_predict.append(DistributionPrediction(configs=self.configs,
                                                seq_len=seq_len,
                                                period_len=period_len,
                                                pred_len=pred_len).to(self.device))
            self.decomps.append(series_decomp(period_len.item() + 1, self.configs))
    def _get_weight(self, x):
        xf = torch.fft.rfft(x, dim=1)
        period_index = x.shape[1] // self.configs.period_list
        amps = abs(xf)
        period_amps = amps[:, period_index, :]
        period_weight = period_amps / period_amps.sum(axis=1, keepdims=True)
        # period_weight = F.softmax(period_amps, dim=1)
        return period_weight
    
    def normalize(self, input):
        if 'adaptive' in self.station_type:
            bs, len, dim = input.shape        
            normalized_input = []
            stat_pred = []
            
            
            for i in range(self.k):
                period_len = self.period_list[i]
                if len % period_len != 0:
                    length = ((len // period_len) + 1) * period_len
                    padding = input[:, -(length - len):, :]
                    input = torch.cat([input, padding], dim=1)
                    
                input_s, input_t = self.decomps[i](input)
                
                norm_input_t, mean_t, std_t = self.norm_t(input_t, bs, len, dim, period_len)
                norm_input_s, mean_s, std_s = self.norm_s(input_s, bs, len, dim, period_len)
                
                # norm_input_s, mean_s, std_s = self.norm_s(input, bs, len, dim, period_len)
                
                normalized_input.append(norm_input_s + norm_input_t)
                stat_pred.append(self.stat_predict[i](input_t, input_s, (mean_t, mean_s), (std_t, std_s)))
                input = input.reshape(bs, -1, dim)[:, :len, :]
            return normalized_input, stat_pred
        else:
            return input, None
    
    
    def norm_t(self, input, bs, len, dim, period_len):
        
        input = input.reshape(bs, -1, period_len, dim)
        mean = torch.mean(input, dim=-2, keepdim=True)
        std = torch.std(input, dim=-2, keepdim=True, unbiased=False)
        norm_input = (input - mean) / (std + self.epsilon)
        norm_input = norm_input.reshape(bs, -1, dim)[:, :len, :]
        
        return norm_input, mean, std
    def norm_s(self, input, bs, len, dim, period_len):
        
        input = input.reshape(bs, -1, period_len, dim).permute(0, 2, 1, 3)
        mean = torch.mean(input, dim=-2, keepdim=True)
        std = torch.std(input, dim=-2, keepdim=True, unbiased=False)
        norm_input = (input - mean) / (std + self.epsilon)
        norm_input = norm_input.permute(0, 2, 1, 3).reshape(bs, -1, dim)[:, :len, :]
        
        return norm_input, mean, std
    def de_normalize(self, input_list, stat_pred, minmax_pred=None, return_outputlist=False):
      
        # print(input_list.shape)
        if 'adaptive' in self.station_type:
            bs, len, dim = input_list[0].shape
            denormalized_output = []
            for i in range(self.k):
                input = input_list[i]
                period_len = self.period_list[i]
                if len % period_len != 0:
                    length = ((len // period_len) + 1) * period_len
                    padding = input[:, -(length - len):, :]
                    input = torch.cat([input, padding], dim=1)
                input_s, input_t = self.decomps[i](input)
                denorm_output_t = self.de_norm_t(input_t, stat_pred[i], bs, len, period_len, dim)
                denorm_output_s = self.de_norm_s(input_s, stat_pred[i], bs, len, period_len, dim)
                # denorm_output_s = self.de_norm_s(input, stat_pred[i], bs, len, period_len, dim)
                
                denormalized_output.append(denorm_output_t + denorm_output_s)
                input = input.reshape(bs, -1, dim)[:, :len, :]
            denormalized_output = torch.stack(denormalized_output, dim=-1)
            self.period_weight = self._get_weight(input).permute(0, 2, 1)
            weight = self.period_weight.unsqueeze(1).repeat(1, len, 1, 1).to(input.device)
            weighted_denorm_output = torch.sum(denormalized_output * weight, -1).reshape(bs, len, dim)
            if return_outputlist:    
                return weighted_denorm_output.float(), denormalized_output
            else:
                return weighted_denorm_output.float()
        else:
            return input_list
    
    def de_norm_t(self, input, stat_pred, bs, len, period_len, dim):
        input = input.reshape(bs, -1, period_len, dim)
        mean = stat_pred[0][:, :, :dim].unsqueeze(2)
        std = stat_pred[0][:, :, dim:].unsqueeze(2)
        denorm_output = input * (std + self.epsilon) + mean
        denorm_output = denorm_output.reshape(bs, -1, dim)[:, :len, :]
        return denorm_output
    
    def de_norm_s(self, input, stat_pred, bs, len, period_len, dim):
        input = input.reshape(bs, -1, period_len, dim).permute(0, 2, 1, 3)
        mean = stat_pred[1][:, :, :dim].unsqueeze(2)
        std = stat_pred[1][:, :, dim:].unsqueeze(2)
        denorm_output = input * (std + self.epsilon) + mean
        denorm_output = denorm_output.permute(0, 2, 1, 3).reshape(bs, -1, dim)[:, :len, :]
        return denorm_output
    
class DistributionPrediction(nn.Module):
    def __init__(self, configs, seq_len, period_len, pred_len):
        super(DistributionPrediction, self).__init__()
        self.seq_len = seq_len
        self.period_len = period_len
        self.pred_len = pred_len
        self.channels = configs.enc_in if configs.features == 'M' else 1
       
        self.model_mean = MLP(input_len=seq_len * period_len, 
                              seq_len=seq_len, 
                              period_len=period_len,
                              pred_len=pred_len, 
                              enc_in=self.channels,
                              mode='mean')
        self.model_std = MLP(input_len=seq_len * period_len, 
                              seq_len=seq_len, 
                              period_len=period_len,
                              pred_len=pred_len, 
                              enc_in=self.channels,
                              mode='std')
        self.model_mean_phase = MLP_phase(input_len=seq_len * period_len, 
                              seq_len=seq_len, 
                              period_len=period_len,
                              pred_len=pred_len, 
                              enc_in=self.channels,
                              mode='mean')
        self.model_std_phase = MLP_phase(input_len=seq_len * period_len, 
                              seq_len=seq_len, 
                              period_len=period_len,
                              pred_len=pred_len, 
                              enc_in=self.channels,
                              mode='std')                          
        self.weight_t = nn.Parameter(torch.ones(2, self.channels)) 
        self.weight_s = nn.Parameter(torch.ones(2, self.channels))
        
    def forward(self, input_t, input_s, mean, std):
        mean_all_t = torch.mean(input_t, dim=1, keepdim=True)
        # outputs_mean = self.model_mean(mean.squeeze(2) - mean_all, input - mean_all) + mean_all 
        outputs_mean_t = self.model_mean(mean[0].squeeze(2) - mean_all_t, input_t - mean_all_t) * self.weight_t[0] + mean_all_t * \
                        self.weight_t[1]
        outputs_std_t = self.model_std(std[0].squeeze(2), input_t)
        outputs_t = torch.cat([outputs_mean_t, outputs_std_t], dim=-1)
        
        mean_all_s = torch.mean(input_s, dim=1, keepdim=True)
        
        outputs_mean_s = self.model_mean_phase(mean[1].squeeze(2) - mean_all_s, input_s - mean_all_s) * self.weight_s[0] + mean_all_s * \
                        self.weight_s[1]
        outputs_std_s = self.model_std_phase(std[1].squeeze(2), input_s)
        outputs_s = torch.cat([outputs_mean_s, outputs_std_s], dim=-1)
        
        return (outputs_t, outputs_s)

class MLP(nn.Module):
    def __init__(self, input_len, seq_len, period_len, pred_len, enc_in, mode):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.period_len = period_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.input_len = input_len
        self.mode = mode

        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.input_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)

class MLP_phase(nn.Module):
    def __init__(self, input_len, seq_len, period_len, pred_len, enc_in, mode):
        super(MLP_phase, self).__init__()
        self.seq_len = seq_len
        self.period_len = period_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.input_len = input_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.period_len, 512)
        self.input_raw = nn.Linear(self.input_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(1024, self.period_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)