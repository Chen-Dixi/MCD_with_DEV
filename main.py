## 自己的
import MCD.functions as functions
from config import get_train_arguments
from MCD.dataloader import PairedDatasetHelper
from MCD.models import *
from MCD.utils import *
## pytorch 
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
## torchvision
from torchvision.transforms import transforms

## dixitool
from dixitool.pytorch.datasets import ImageList
#from dixitool.pytorch.datasets import PairedDatasetHelper
from dixitool.metric import AccCalculatorForEveryClass
from dixitool.pytorch.module import functional as dixiF
# other package
import random
from tqdm import tqdm
import os




#都可以用的就放里面，在另外定制的加外面
parser = get_train_arguments()
#get_train_arguments里面就放必要的arguments
#接下来 parser里面可以加一些别的东西
#parser.add_argument('--checkpoint-file-name', default='step2checkpoint.pth.tar', type=str, metavar='checkpoint-file-name',
#  help='name for checkpoint files') #只给文件名，一般后面的处理都会在path前面加一个checkpoints文件夹
parser.add_argument('--num-k', type=int, default=4, metavar='K',help='how many steps to repeat the generator update')
parser.add_argument('--dataset-prefix',type=str,default='',help="prefix for dataset list file if needed") #/User/chendixi/Datasets/VisDA2017/
parser.add_argument('--train-path', type=str, default=None, metavar='B',required=True,
                        help='image_list.txt of train split')
parser.add_argument('--validation-path', type=str, default=None,required=True,
                        help='image_list.txt of validation split')
parser.add_argument('--num-layer', type=int, default=3, metavar='K',
                        help='how many layers for classifier')
#设置，所以叫option
opt = parser.parse_args()

opt=functions.post_config(opt) #给opt加一些 argparse加不了的东西
opt.class_names = functions.generate_class_names(opt)
functions.print_config(opt) #打印 parser.parse_opt()里面的东西
#===== 创建用来保存模型的文件夹
functions.prepare_dir2save_folder(opt.dir2save)

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

## dataset
synthetic_dataset = ImageList(opt.train_path,os.path.join(opt.dataset_prefix,'train'), transform=data_transforms["train"])
realistic_dataset = ImageList(opt.validation_path, os.path.join(opt.dataset_prefix,'validation'), transform=data_transforms["val"])

datasets_helper = PairedDatasetHelper(synthetic_dataset,realistic_dataset,opt.batch_size,shuffle=True) #可以有len
# this dataloader contain 'train' split and 'validation' split
train_dataLoader = datasets_helper.load_data()

test_dataLoader = DataLoader(realistic_dataset,batch_size=opt.batch_size,shuffle=False)#没有

# 模型，一个G，两个个F
G = ResBase(opt.net).to(opt.device)
F1 = ResClassifier(num_layer=opt.num_layer).to(opt.device)
F2 = ResClassifier(num_layer=opt.num_layer).to(opt.device)
F1.apply(weights_init)
F2.apply(weights_init)

# 优化器Optimizer
# 暂时提供SGD
optimizer_g = optim.SGD(G.features.parameters(), lr=opt.lr,weight_decay=0.0005)
optimizer_f = optim.SGD([{'params':F1.parameters()},{'params':F2.parameters()}],momentum=opt.momentum,lr=opt.lr,weight_decay=0.0005)

ce_criterion = nn.CrossEntropyLoss()

accMetric = AccCalculatorForEveryClass( opt.num_classes )
accMetric.set_classes_name(opt.class_names)
accMetric.set_best_method(best_method='total_acc')
accMetric.set_header_info()

def discrepancy( out1, out2):
        return torch.mean(torch.abs(F.softmax(out1,dim=1) - F.softmax(out2,dim=1)))


def train(G,F1,F2, optimizer_g, optimizer_f, train_dataLoader,opt):
    G.train()
    F1.train()
    F2.train()
    
    epoch_src_loss = 0.0
    epoch_entropy_loss = 0.0
    epoch_discrepancy_loss = 0.0

    batch = next(iter(train_dataLoader))
    src_data, src_target = batch['s'], batch['s_target']
    tgt_data, tgt_target = batch['t'], batch['t_target']

    src_data, src_target = src_data.to(opt.device), src_target.to(opt.device)
    tgt_data, tgt_target = tgt_data.to(opt.device), tgt_target.to(opt.device)

    

    for idx , batch in enumerate(tqdm(train_dataLoader)):
        src_data, src_target = batch['s'], batch['s_target']
        tgt_data, tgt_target = batch['t'], batch['t_target']

        src_data, src_target = src_data.to(opt.device), src_target.to(opt.device)
        tgt_data, tgt_target = tgt_data.to(opt.device), tgt_target.to(opt.device)

        s_feature = G(src_data)
        t_feature = G(tgt_data)

        s_output1 = F1(s_feature)
        s_output2 = F2(s_feature)
        t_output1 = F1(t_feature)
        t_output2 = F2(t_feature)

        # stepA
        entropy_loss = - torch.mean(torch.log(torch.mean(F.softmax(t_output1,dim=1),0)+1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(F.softmax(t_output2,dim=1),0)+1e-6))

        s_loss1 = ce_criterion( s_output1, src_target)
        s_loss2 = ce_criterion( s_output2, src_target)

        loss = s_loss1 + s_loss2 + 0.01 * entropy_loss

        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        #Step B train classifier to maximize discrepancy
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        #StepB
        s_feature = G(src_data)
        t_feature = G(tgt_data)

        s_output1 = F1(s_feature)
        s_output2 = F2(s_feature)
        t_output1 = F1(t_feature)
        t_output2 = F2(t_feature)
        
        s_loss1 = ce_criterion( s_output1, src_target)
        s_loss2 = ce_criterion( s_output2, src_target)

        entropy_loss = - torch.mean(torch.log(torch.mean(F.softmax(t_output1,dim=1),dim=0)+1e-6))
        entropy_loss -= torch.mean(torch.log(torch.mean(F.softmax(t_output2,dim=1),dim=0)+1e-6))
        loss_dis = discrepancy(t_output1,t_output2)
        loss = s_loss1 + s_loss2 + 0.01*entropy_loss - 1.0*loss_dis

        loss.backward()
        optimizer_f.step()

        epoch_src_loss += (s_loss1.item()+s_loss2.item())     
        epoch_entropy_loss += entropy_loss.item()
        epoch_discrepancy_loss += loss_dis.item()
        #Step C
        for i in range(opt.num_k):
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()

            t_feature = G(tgt_data)
            t_output1 = F1(t_feature)
            t_output2 = F2(t_feature)
            loss_dis = discrepancy(t_output1,t_output2)
            loss_dis.backward()
            optimizer_g.step()


    
    
    
    print("src_loss: %f" %epoch_src_loss)
    print("entropy_loss : %f" % epoch_entropy_loss)
    print("discrepancy_loss : %f" % epoch_discrepancy_loss)


best_acc_total = 0.0

def test(G, F1, F2):
    global best_acc_total
    G.eval()
    F1.eval()
    F2.eval()

    accMetric.reset()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_dataLoader)):
            v_data, v_target = functions.batch2data(batch, opt.device)
        
            feature = G(v_data)
            output1 = F1(feature)
            output2 = F2(feature)
            accMetric.update(output1,v_target)
            accMetric.update(output2,v_target)

    acc_total = accMetric.get()
    is_best = acc_total > best_acc_total
    best_acc_total = max(acc_total,best_acc_total)

    print("Test accuracy: %.4f %" % acc_total)
    dixiF.save_checkpoint({
            'best_acc_total': best_acc_total,
            'acc_total':acc_tota,
            'netG':G,
            'netF1':F1,
            'netF2':F2,
        }, is_best, opt.dir2save,filename='checkpoint.pth.tar')

try:
    for epoch in range(1,2):#opt.epochs+1):
        train(G,F1,F2, optimizer_g, optimizer_f,train_dataLoader ,opt)
        if epcoh % opt.test_interver == 0:
            test(G,F1,F2)
        
        accMetric.step(epoch)

except KeyboardInterrupt:
    print("error")
    
else:
    print("finish")
    



        

        





