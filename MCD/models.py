from dixitool.pytorch.module import GradReverseLayer

#torchvision
from torchvision import models

# pytorch
import torch.nn as nn
import torch.nn.functional as F
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
        x = x.view(x.size(0), self.dim)
        return x

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
# 用卷积来下采样

# class DomainDiscriminator(nn.Module):

#     def __init__(self,opt):
#         super(DomainDiscriminator, self).__init__()
#         self.in_dim = 2048
#         if opt.net=='resnet18':
#             self.in_dim = 512

#         # self.fc1 = nn.Linear(50 * 4 * 4, 100)
#         # self.bn1 = nn.BatchNorm1d(100)
#         # self.fc2 = nn.Linear(100, 2)
#         self.fc1 = nn.Linear(self.dim, 100)
#         # one nerve cell
#         self.fc2 = nn.Linear(100, 1)

#     def forward(self, x):
#         # input = GradientReversalLayer.grad_reverse(input, lambda_p)
#         # logits = F.relu(self.bn1(self.fc1(input)))
#         # logits = F.log_softmax(self.fc2(logits), 1)
#         logits = F.relu(self.fc1(x))
#         logits = torch.sigmoid(self.fc2(x))

#         return logits
class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, ker_size = 4, stride=2,padding=1 ):
        super(ConvBlock,self).__init__()
        self.add_module("conv", nn.Conv2d(in_channels, out_channels,ker_size,stride,padding))
        sefl.add_module("norm", nn.BatchNorm2d(out_channels))
        self.add_module('LeakyReLU', nn.LeakyReLU(0.2,inplace=True))


class DomainDiscriminator(nn.Module):

    # 图片输入 input is (3) x 224 x 224
    def __init__(self, in_channels=3, ndf=64):
        
        N = int(ndf)
        self.head = ConvBlock(in_channels, N) 
        self.body = nn.Sequential()
        
        for i in range(4):
            N=int(ndf/pow(2,(i+1)))
            block = ConvBlock(max(32,2*N),max(32,N))
            self.body.add_module("block%d"%(i+1),block)

        self.tail = ConvBlock(max(32,N),1,7,stride=1,padding=0)

    def forward(x):
        x = self.head(x) # 224 x 224
        x = self.body(x) # 7 x 7
        x = self.tail(x) # 1 x 1  size(size(0),1,1,1)
        x = F.sigmoid(x)
        x = x.view(x.size(0), -1)
        return x

        





