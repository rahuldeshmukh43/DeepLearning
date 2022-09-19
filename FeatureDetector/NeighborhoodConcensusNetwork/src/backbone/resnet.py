import torch.nn as nn
import importlib

class ResNet(nn.Module):
    def __init__(self,
                 encoder='resnet101',
                 pretrained=True):        
        super(ResNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type" 
        resnet = class_from_name("torchvision.models", encoder)(pretrained=pretrained)
        self.firstconv = resnet.conv1  # H/2
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool  # H/4
        # encoder
        self.layer1 = resnet.layer1  # H/4
        self.layer2 = resnet.layer2  # H/8
        self.layer3 = resnet.layer3  # H/16
        
    def forward(self, x):
        x = self.firstrelu(self.firstbn(self.firstconv(x)))
        x = self.firstmaxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x       

def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)