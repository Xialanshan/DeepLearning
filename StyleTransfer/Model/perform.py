import torch
import torch.nn as nn
import torch.optim as optim
import time
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models 
from module import ContentLoss, StyleLoss, TotalVariationLoss, Normalization, cnn_normalization_mean, cnn_normalization_std

# device: GPU, version: torch-2.2.1+cu118 torchaudio-2.2.1+cu118 torchvision-0.17.1+cu118 typing-extensions-4.8.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

"""
图像加载: 选择图片分辨率
"""
content_img = Image.open("./Image_mini/dancing.jpg")
style_img = Image.open("./Image_mini/style2.jpg")

imsize = 512
loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),
    transforms.ToTensor()
])  # 重置任何被加载的图像大小: 512*512, 并转换成张量

def image_to_Tenser(image_name):
    image = loader(image_name).unsqueeze(0)      # 添加批处理维度
    return image.to(device, torch.float)
style_img = image_to_Tenser(style_img)
content_img = image_to_Tenser(content_img)

assert style_img.size()==content_img.size(), \
"we need to import style and content images of the same size"

"""
图像显示
"""
plt.ion()   # 开启matplotlib的交互模式

unloader = transforms.ToPILImage()
def tensor_to_image(tensor, title=None):
    image = tensor.cpu().clone()    # plt在cpu上显示
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title != None:
        plt.title(title)
    plt.pause(0.001)



content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
tv_layers_default = ['conv_5']


"""
获取风格迁移模型, 计算风格损失和内容损失和总变差损失
"""

inception = models.inception_v3(pretrained=True).eval()     # 提取模型的特征提取部分,并且设置成评估模式,固定权重
def get_style_model_losses(inception, normalization_mean, normalization_std,
                           style_img, content_img,
                           content_layers=content_layers_default,
                           style_layers=style_layers_default,
                           tv_layers = tv_layers_default):
    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []
    tv_losses = []

    model = nn.Sequential(normalization)
    i = 0
    inception_children = nn.Sequential(*list(inception.children())[:7])
    for layer in inception_children.children():
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        else:
            i += 1
            name = 'conv_{}'.format(i)
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()       
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)     
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

        if name in tv_layers:
            tv_loss = TotalVariationLoss()
            model.add_module("tv_loss_{}".format(i), tv_loss)
            tv_losses.append(tv_loss)

    for i in range(len(model)-1, -1, -1):
        if  isinstance(model[i], TotalVariationLoss):
            break

    model = model[:(i + 1)]     
    return model, style_losses, content_losses, tv_losses




"""
输入图像处理
"""
input_img = content_img.clone()

def get_input_optimizer(input_img):     
    optimizer = optim.LBFGS([input_img])    # 拟牛顿法变种优化
    return optimizer


"""
风格迁移
"""
def run_style_transfer(inception, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000000, content_weight=100, tv_weight=0.0000001):    
    print("Building the style transfer model..")
    model, style_losses, content_losses, tv_losses = get_style_model_losses(inception, normalization_mean, normalization_std,
                           style_img, content_img)
    # print(model)
    input_img.requires_grad_(True)      
    model.eval()
    model.requires_grad_(True)

    optimizer = get_input_optimizer(input_img)

    print("Optimizing..")
    
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)    
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            tv_score = 0
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            for tvl in tv_losses:
                tv_score += tvl.loss
            style_score *= style_weight
            content_score *= content_weight
            tv_score *= tv_weight
            loss = style_score + content_score + tv_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} | Content Loss: {:4f} | TotalVariation Loss: {:4f}'.format(
                    style_score.item(), content_score.item(), tv_score.item()))
                print()

            return style_score + content_score
        
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0,1)
    
    return input_img


start_time = time.time()
output = run_style_transfer(inception, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)
end_time = time.time()

run_time = end_time - start_time
print(f"代码段的运行时间：{run_time} 秒")

plt.figure()
tensor_to_image(output, title='Output Image')

plt.ioff()
plt.show()


