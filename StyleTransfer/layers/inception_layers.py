import torchvision.models as models
import torch.nn as nn

inception = models.inception_v3(pretrained=True).eval()

model = nn.Sequential()
inception = nn.Sequential(*list(inception.children())[:7])

layer_count = {}
for name, child in inception.named_children():
    layer_type = str(child.__class__).split(".")[-1][:-2]       
    if layer_type in layer_count:
        layer_count[layer_type] += 1
    else:
        layer_count[layer_type] = 1


for layer_type, count in layer_count.items():
    print(f"Layer Type: {layer_type}, Count: {count}")

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
tv_layers_default = ['conv_5']

i = 0
for layer in inception.children():
    if isinstance(layer, nn.MaxPool2d):
        name = 'pool_{}'.format(i)
    else:
        i += 1
        name = 'conv_{}'.format(i)
    model.add_module(name, layer)

print(model)