# MCD DEV
from config import get_train_arguments
from MCD.dataloader import PairedDatasetHelper
from MCD.models import DomainDiscriminator
import MCD.functions as functions
#dixitool
from dixitool.pytorch.datasets import ImageList

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
# --lr 0.0005



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

if __name__ == '__main__':

    parse = get_train_arguments()
    parse.add_argument('--ndf',type=int,default=64)
    parser.add_argument('--validation-path', type=str, default='/home/chendixi/Datasets/VisDA2017/validation/image_list.txt',required=True,
                            help='image_list.txt of validation split')
    parser.add_argument('--test-path', type=str, default='/home/chendixi/Datasets/VisDA2017/test/image_list_label.txt',required=True,
                            help='image_list.txt of test split')   
    parser.add_argument('--dataset-prefix',type=str,default='/home/chendixi/Datasets/VisDA2017',help="prefix for dataset list file if needed")                     

    

    opt = parse.parse_args()
    opt = post_config(opt)

    functions.prepare_dir2save_folder(opt.dir2save)


    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    validation_dataset = ImageList(opt.validation_path, os.path.join(dataset_prefix,'validation'), transform=transform)
    test_dataset = ImageList(opt.test_path, os.path.join(dataset_prefix,'test'), transform=transform)

    datasets_helper = PairedDatasetHelper(validation_dataset,test_dataset,opt.batch_size,shuffle=False)
    train_dataLoader = datasets_helper.load_data()

    netD = DomainDiscriminator(ndf=opt.ndf).to(opt.device)

    optimizer = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    criterion = nn.BCELoss()
    
    for epoch in range(1,opt.epochs+1):
        train_loss = 0.0
        for idx, batch in enumerate(tqdm(train_dataLoader)):
            src_data = batch['s'] # validation
            tgt_data = batch['t'] # test

            src_data = src_data.to(opt.device)
            tgt_data = tgt_data.to(opt.device)

            optimizer.zero_grad()
            out = netD(src_data).view(-1) # (batch,)
            label = torch.full((src_data.size(0),), 1, device=device)
            err_validation = criterion(out, label)
            err_validation.backward()

            

            out = netD(tgt_data).view(-1)

            label = torch.full((tgt_data.size(0),), 0, device=device)
            err_test = criterion(out, label)
            err_test.backward()

            optimizer.step()

            err_D = err_test+err_validation
            train_loss += err_D.item()

        print("train loss: %f"%(train_loss))

        torch.save(netD.state_dict(), '%s/netD.pth' % (opt.dir2save))



