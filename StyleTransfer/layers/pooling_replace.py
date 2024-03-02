import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练的VGG-19模型
cnn = models.vgg19(pretrained=True).features.eval()

# 自定义替换最大池化层为平均池化层
def replace_maxpool_with_avgpool(model):       
    for child_name, child in model.named_children():
        if isinstance(child, nn.MaxPool2d):
            setattr(model, child_name, nn.AvgPool2d(kernel_size=2, stride=2))

replace_maxpool_with_avgpool(cnn)

print(cnn)


"""
实际效果: 平均池化层效果远不如最大池化层效果, 故不替换
"""
