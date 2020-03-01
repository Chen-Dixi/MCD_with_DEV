from dixitool.pytorch.module import GradReverseLayer

#torchvision
from torchvision import models

# pytorch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
        

class ResBase(nn.Module):
    def __init__(self,net='resnet18',pret=True):
        super(ResBase, self).__init__()
        self.dim = 2048
        if net == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if net == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if net == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if net == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)

        mod = list(model_ft.children())
        mod.pop()
        #self.model_ft =model_ft
        self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), slef.dim)

class ResClassifier(nn.Module):
    def __init__(self, num_classes=12,num_layer = 3,num_unit=2048,prob=0.5,middle=1000):
        super(ResClassifier, self).__init__()
        layers = []
        self.grad_reverse_layer = GradReverseLayer()
        # currently 10000 units
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit,middle))
        layers.append(nn.BatchNorm1d(middle,affine=True))
        layers.append(nn.ReLU(inplace=True))


        for i in range(num_layer-2):
            layers.append(nn.Dropout(p=prob))  
            layers.append(nn.Linear(middle,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle,num_classes))
        self.classifier = nn.Sequential(*layers)

        #self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(2048, 1000),
        #    nn.BatchNorm1d(1000,affine=True),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(1000, 1000),
        #    nn.BatchNorm1d(1000,affine=True),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(1000, num_classes),

    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x,reverse=False):
        if reverse:
            x = self.grad_reverse_layer(x, self.lambd)
        x = self.classifier(x)
        return x

# 领域判别器 Domain Discriminator

class DomainDiscriminator(nn.Module):
    def __init__(self,opt,feature_dim=2048):



