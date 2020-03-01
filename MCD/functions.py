import torch
import random
import numpy as np 
import os

def print_dict(args):
    
    for arg, content in args.__dict__.items():
        print("{}:{}".format(arg, content))
    print("\n")

def print_config(args):
    print("==========================================")
    print("==========       CONFIG      =============")
    print("==========================================")
    print_dict(args)
    
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
    opt.dir2save = generate_dir2save(opt)
    opt.class_names = generate_class_names(opt)
    opt.num_classes = generate_num_classes(opt)
    return opt

def generate_dir2save(opt):
    dir2save = "TrainedModel/%s/num_k=%d" % (opt.net, opt.num_k)
    return dir2save

def prepare_dir2save_folder(dir2save):
    #先生成要保存的地址

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass



def generate_class_names(opt):
    if opt.dataset=="visda":
        return ["aeroplane","bicycle","bus","car","horse","knife",
                "motorcycle","person","plant","skateboard","train","truck"]

    return ""

def generate_num_classes(opt):
    if opt.dataset=="visda":
        return 12
    else:
        raise NotImplementedError

def save_networks(net_G,net_F1, net_F2, opt):
    torch.save(net_G, '%s/netG.pth' % (opt.dir2save))
    torch.save(net_F1, '%s/net_F1.pth' % (opt.dir2save))
    torch.save(net_F2, '%s/net_F2.pth' % (opt.dir2save))

def batch2data(batch, device):

    data, target = batch
    return data.to(device), target.to(device)