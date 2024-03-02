import torch.nn as nn
import torchvision.models as models

cnn = models.vgg19(pretrained=True).classifier.eval()

# 统计层名和层数
layer_count = {}
for name, child in cnn.named_children():
    layer_type = str(child.__class__).split(".")[-1][:-2]  
    if layer_type in layer_count:
        layer_count[layer_type] += 1
    else:
        layer_count[layer_type] = 1


for layer_type, count in layer_count.items():
    print(f"Layer Type: {layer_type}, Count: {count}")

model = nn.Sequential()
i = 0
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        name = 'conv_{}'.format(i)
    elif isinstance(layer, nn.ReLU):
        name = 'relu_{}'.format(i)
        layer = nn.ReLU(inplace=False)
    elif isinstance(layer, nn.MaxPool2d):
        name = 'pool_{}'.format(i)
    elif isinstance(layer, nn.BatchNorm2d):
        name = 'bn_{}'.format(i)
    else:
        raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

    model.add_module(name, layer)
print(model)