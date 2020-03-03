import argparse

def get_train_arguments():
    parser = argparse.ArgumentParser(description="Visda2017 MCD")
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=None, metavar='N',
                        help='if none then test-batch-szie = --batch-size (default: None)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument('--not-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--seed', type=int, default=19950907, metavar='S',
                        help='random seed (default: birthday)')
    parser.add_argument('--test-interval', type=int, default=4, metavar='N',
                        help='how many epochs to wait before test')
    parser.add_argument('--name', type=str, default='board', metavar='B',
                        help='board dir')
    parser.add_argument('--save', type=str, default='save/mcd', metavar='B',
                        help='board dir')
    parser.add_argument('--net', type=str, default='resnet50', metavar='B',
                        help='which resnet 18,50,101,152,200')
    parser.add_argument('--build',type=int,default=0,help="the number of build")
    return parser

def get_inference_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--not-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--cuda',type=int,default=0,metavar='cuda')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--imagenet-pretrained',type=bool, default=True,metavar='N',help='imagenet pretrained option(default: True)')
    parser.add_argument('--seed', type=int, default=19950907, metavar='S',
                        help='random seed (default: 19950907) my birthday')
    parser.add_argument('--net', type=str, default='resnet50', metavar='B',
                        help='which network alex,vgg,res?')

    return parser

