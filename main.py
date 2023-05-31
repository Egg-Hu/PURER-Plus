import argparse
import os
import torch
from method.purer import PURER
from tool import setup_seed,pretrains

parser = argparse.ArgumentParser(description='purer-plus')
#basic
parser.add_argument('--multigpu', type=str, default='0', help='seen gpu')
parser.add_argument('--gpu', type=int, default=0, help="gpu")
parser.add_argument('--dataset', type=str, default='cifar100', help='cifar100/miniimagenet')
parser.add_argument('--testdataset', type=str, default='cifar100', help='cifar100/miniimagenet')
parser.add_argument('--val_interval',type=int, default=2000)
parser.add_argument('--save_interval',type=int, default=2000)
parser.add_argument('--episode_batch',type=int, default=4)
#maml
parser.add_argument('--way_train', type=int, default=5, help='way')
parser.add_argument('--num_sup_train', type=int, default=5)
parser.add_argument('--num_qur_train', type=int, default=15)
parser.add_argument('--way_test', type=int, default=5, help='way')
parser.add_argument('--num_sup_test', type=int, default=5)
parser.add_argument('--num_qur_test', type=int, default=15)
parser.add_argument('--backbone', type=str, default='conv4', help='conv4')
parser.add_argument('--episode_train', type=int, default=240000)
parser.add_argument('--episode_test', type=int, default=600)
parser.add_argument('--start_id', type=int, default=1)
parser.add_argument('--inner_update_num', type=int, default=5)
parser.add_argument('--test_inner_update_num', type=int, default=10)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--outer_lr', type=float, default=0.001)
parser.add_argument('--approx', action='store_true',default=False)
#method
parser.add_argument('--method', type=str, default='purer', help='purer')
parser.add_argument('--teacherMethod', type=str, default='protonet', help='anil/maml/protonet')
#dfmeta
parser.add_argument('--inversionMethod', type=str, default='deepinv')
parser.add_argument('--way_pretrain', type=int, default=5, help='way')
parser.add_argument('--APInum', type=int, default=100)
parser.add_argument('--pre_backbone', type=str, default='conv4', help='conv4/resnet10/resnet18')
parser.add_argument('--noBnLoss', action='store_true',default=False)
parser.add_argument('--Glr', type=float, default=0.001)
parser.add_argument('--candidate_size', type=int, default=13)

#else
parser.add_argument('--extra', type=str, default='')
parser.add_argument('--pretrained_prefix', type=str, default='./pretrained')

#pretrain
parser.add_argument('--pretrain', action='store_true',default=False)
parser.add_argument('--unique', action='store_true',default=False)

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.multigpu
setup_seed(2022)
args.device=torch.device('cuda:{}'.format(args.gpu))
########
if args.pretrain:
    pretrains(args,args.APInum)
    #pretrains_unique(args)
else:
    method_dict = dict(
                purer=PURER,

    )
    method=method_dict[args.method](args)
    method.train_loop()
