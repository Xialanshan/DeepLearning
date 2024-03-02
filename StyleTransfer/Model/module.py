import torch
import torch.nn as nn
import torch.nn.functional as F     

class ContentLoss(nn.Module):       
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        self.target = target.detach()       # 内容图像表征剥离

    def forward(self,input):
        self.loss = F.mse_loss(input, self.target)  # 输入图像和内容图像之间的均方误差
        return input
    

def gram_matrix(input):                     # 捕捉不同通道之间的特征相关性
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())    # 特征向量的点积矩阵
    return G.div(a * b * c * d)             # 标准化, 确保计算结果不受输入图像尺寸的影响

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()      # 目标风格图像的Gram矩阵

    def forward(self, input):
        G = gram_matrix(input)      # 输入图像的Gram矩阵
        self.loss = F.mse_loss(G,self.target)
        return input
    

"""addtional loss"""
class TotalVariationLoss(nn.Module):        # 平滑图像
    def __init__(self):
        super(TotalVariationLoss,self).__init__()
        self.loss = 0  

    def forward(self, input):
        h_tv = torch.sum((input[:, :, 1:, :] - input[:, :, :-1, :]).abs())
        w_tv = torch.sum((input[:, :, :, 1:] - input[:, :, :, :-1]).abs())
        self.loss =  h_tv + w_tv
        return input
    

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])


class Normalization(nn.Module):     
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self, img):
        return (img - self.mean) / self.std
    


