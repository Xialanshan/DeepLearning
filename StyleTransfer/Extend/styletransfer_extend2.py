import torch
import torch.nn as nn
import torch.nn.functional as F     
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import torchvision.models as models 

class ContentLoss(nn.Module):
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        self.target = target.detach()

    def forward(self,input):
        self.loss = F.mse_loss(input, self.target)  
        return input
    

def gram_matrix(input):     
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())    
    return G.div(a * b * c * d)             

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G,self.target)
        return input
    

"""addition"""
class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()
        self.loss = 0  

    def forward(self, input):
        h_tv = torch.sum((input[:, :, 1:, :] - input[:, :, :-1, :]).abs())
        w_tv = torch.sum((input[:, :, :, 1:] - input[:, :, :, :-1]).abs())
        self.loss = h_tv + w_tv  
        return self.loss  


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):     
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1,1,1)
        self.std = torch.tensor(std).view(-1,1,1)

    def forward(self, img):
        return (img - self.mean) / self.std
    

# device: GPU, version: 2.1.0+cu118
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


print("请输入您喜欢的画家(序号): ")
print("1:Vermeer    2:莫奈    3:Redouté    4:齐白石    5: 梵高")
print("-"*60)
painter_index = input()
painter = {'1':"Johannes_Vermeer", '2':"Monet", 
           '3':"Pierre_Joseph", '4':"QiBaishi", '5':"Van_Gogh"}
painter_index = painter[painter_index]
paint_index = random.randint(1, 5)


imsize = 512
loader = transforms.Compose([
    transforms.Resize((imsize,imsize)),
    transforms.ToTensor()
])  

def image_to_Tensor(image_name):
    image = Image.open(image_name)
    image = image.convert("RGB")
    image = loader(image).unsqueeze(0)      
    return image.to(device, torch.float)


style_img = image_to_Tensor(f"./Image/{painter_index}/{paint_index}.jpg")
content_img = image_to_Tensor("./Image_mini/dancing.jpg")


assert style_img.size()==content_img.size(), \
"we need to import style and content images of the same size"


plt.ion()   
unloader = transforms.ToPILImage()
def tensor_to_image(tensor, title=None):
    image = tensor.cpu().clone()    
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title != None:
        plt.title(title)
    plt.pause(0.001)


inception = models.inception_v3(pretrained=True).eval()    


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
tv_layers_default = ['conv_5']


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
    optimizer = optim.LBFGS([input_img])    # 拟牛顿法变种
    return optimizer


"""
风格迁移
"""
def run_style_transfer(inception, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=500,
                       style_weight=1000000000, content_weight=100, tv_weight=0.0000001):    
    print("Building the style transfer model..")
    model, style_losses, content_losses, tv_losses = get_style_model_losses(inception, normalization_mean, normalization_std,
                           style_img, content_img)
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



output = run_style_transfer(inception, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img)

plt.figure()
tensor_to_image(output, title='Output Image')

plt.ioff()
plt.show()
