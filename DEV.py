#DEV
from config import get_inference_arguments
from MCD.dataloader import PairedDatasetHelper
from MCD.models import DomainDiscriminator, weights_init, ResBase, ResClassifier
import MCD.functions as functions
#dixitool
from dixitool.pytorch.datasets import ImageList
from dixitool.metric import AccCalculatorForEveryClass
#torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
import torch.optim as optim
import torch.nn as nn
#tool
import random
import numpy as np
from tqdm import tqdm
import os
# density_ratio = important_weighted(x_val) #M
# wl = density_ratio * ((predict_source - y_val) ** 2) #L
# err_iwcv = np.mean( wl )

# #计算DEV 度量
# # 计算\eta - cov/var
# cov = np.cov(np.concatenate((wl, density_ratio),axis = 1 ), rowvar=False)[0][1] 
# #ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population.
# #ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
# var = np.var(density_ratio, ddof=1) 

# eta = - cov / var
# err_dev = err_iwcv + eta*np.mean(density_ratio) - eta

def post_config(opt):
    opt.device = torch.device('cuda:%s'%(opt.cuda) if torch.cuda.is_available() else "cpu")
    if opt.seed is None:
        opt.seed = random.randint(1, 10000)
    
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if not opt.not_cuda:
        torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    #输出文件
    opt.dir2save = "TrainedModel/netD"
    return opt
    
def important_weighted(netD, val_data, constant):
    with torch.no_grad():
        output = netD(val_data)
        output = torch.sigmoid(output)
        #constant = (float(n_val) / float(n_test))
        W = ((1-output) / output) * float(constant)

    return W.view(-1)



if __name__ == '__main__':
    
    parse = get_inference_arguments()
    parse.add_argument('--ndf',type=int,default=64)
    parse.add_argument('--validation-path', type=str, default='/home/chendixi/Datasets/VisDA2017/validation/image_list.txt',
                            help='image_list.txt of validation split')
    parse.add_argument('--test-path', type=str, default='/home/chendixi/Datasets/VisDA2017/test/image_list_label.txt',
                            help='image_list.txt of test split')   
    parse.add_argument('--dataset-prefix',type=str,default='/home/chendixi/Datasets/VisDA2017',help="prefix for dataset list file if needed")                     
    parse.add_argument('--netD',type=str,default='',required=True)
    parse.add_argument('--checkpoint-path',type=str,default='',help="checkpoint of MCD")                     
    parse.add_argument('--num-layer', type=int, default=3, metavar='K',
                        help='how many layers for classifier')
    opt = parse.parse_args()
    opt = post_config(opt)
    functions.print_config(opt)
    #dataLoader
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    validation_dataset = ImageList(opt.validation_path, os.path.join(opt.dataset_prefix,'validation'), transform=transform)
    validation_dataLoader = DataLoader(validation_dataset,batch_size=opt.batch_size,shuffle=False)

    test_dataset = ImageList(opt.test_path, os.path.join(opt.dataset_prefix,'test'), transform=transform)

    constant = float(len(validation_dataset)/len(test_dataset))

    netD = DomainDiscriminator(ndf=opt.ndf).to(opt.device)
    netG = ResBase(opt.net).to(opt.device)
    F1 = ResClassifier(num_layer=opt.num_layer).to(opt.device)
    F2 = ResClassifier(num_layer=opt.num_layer).to(opt.device)
    if(opt.netD=='' or opt.checkpoint_path==''):
        raise RuntimeError("请指定模型地址")

    dicts = functions.load_from_checkpoint(opt.checkpoint_path,'netG','netF1','netF2')
    # 加载所有模型参数
    netD.load_state_dict(torch.load(opt.netD))
    netG.load_state_dict(dicts[0])
    #分类器就使用F1
    F1.load_state_dict(dicts[1])
    F2.load_state_dict(dicts[2])

    ce_criterion = nn.CrossEntropyLoss(reduction='none')
    L = np.empty(0,dtype=np.float32)
    W = np.empty(0,dtype=np.float32)
    with torch.no_grad():
        for idx , batch in enumerate(tqdm(validation_dataLoader)):
            val_data, val_target = functions.batch2data(batch,opt.device)

            val_feature = netG(val_data)
            val_out = F1(val_feature)

            batch_err_loss = ce_criterion(val_out, val_target) #size(batch,1)
            #print("err_loss size:")
            #print(batch_err_loss.size()) #size(32,)

            batch_W = important_weighted(netD, val_data, constant) #size(batch,1)
            #print("important_weighted (W) size:")
            #print(batch_W.size()) #size(32,)

            batch_L = batch_W * batch_err_loss
            #print(batch_L.size()) #(32,)
            batch_L_np = batch_L.cpu().numpy()
            batch_W_np = batch_W.cpu().numpy()

            L = np.concatenate((L,batch_L_np),axis=0)
            W = np.concatenate((W,batch_W_np),axis=0)
            #print(batch_W_np.dtype) #float32
            #print(batch_L_np.shape) #(32,)

    L = L.reshape((L.shape[0],1))
    W = W.reshape((L.shape[0],1))

    cov = np.cov(np.concatenate((L, W),axis = 1 ), rowvar=False)[0][1] 
    #ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population.
    #ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
    var = np.var(W, ddof=1) 

    eta = - cov / var
    err_dev = np.mean(L) + eta*np.mean(W) - eta

    print("本次测试针对test split 的 Deep Embedded Validation Target无偏估计是{:f}".format(err_dev))




    





