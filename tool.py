import math
import os.path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.cifar100 import Cifar100, Cifar100_Specific
from dataset.samplers import CategoriesSampler
from dataset.miniimagenet import MiniImageNet, MiniImageNet_Specific
import network
from dataset.cub import CUB, CUB_Specific
from dataset.flower import flower_Specific, flower
from logger import get_logger
from network import Conv4, ResNet34, ResNet18, ResNet50, ResNet10
import torch.nn.functional as F
from torchvision import transforms


def get_dataloader(args,noTransform_test=False,resolution=32):
    if args.dataset=='cifar100':
        trainset = Cifar100(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
    elif args.dataset=='miniimagenet':
        trainset = MiniImageNet(setname='meta_train', augment=False, resolution=resolution)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
    elif args.dataset=='cub':
        trainset = CUB(setname='meta_train', augment=False, resolution=resolution)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
    elif args.dataset=='flower':
        trainset = flower(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
    if args.testdataset == 'cifar100':
        trainset = Cifar100(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size=trainset.img_size
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset=Cifar100(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                          600,
                                          args.way_test,
                                          args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                  num_workers=0,
                                  batch_sampler=val_sampler,
                                  pin_memory=True)
        testset = Cifar100(setname='meta_test', augment=False,noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                num_workers=8,
                                batch_sampler=test_sampler,
                                pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset == 'miniimagenet':
        trainset = MiniImageNet(setname='meta_train', augment=False,resolution=resolution)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset = MiniImageNet(setname='meta_val', augment=False,resolution=resolution)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = MiniImageNet(setname='meta_test', augment=False, noTransform=noTransform_test,resolution=resolution)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset=='cub':
        trainset = CUB(setname='meta_train', augment=False,resolution=resolution)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
        train_sampler = CategoriesSampler(trainset.label,
                                          args.episode_train,
                                          args.way_train,
                                          args.num_sup_train + args.num_qur_train)
        train_loader = DataLoader(dataset=trainset,
                                  num_workers=8,
                                  batch_sampler=train_sampler,
                                  pin_memory=True)
        valset = CUB(setname='meta_val', augment=False,resolution=resolution)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = CUB(setname='meta_test', augment=False, noTransform=noTransform_test,resolution=resolution)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return train_loader, val_loader, test_loader
    elif args.testdataset=='flower':
        trainset = flower(setname='meta_train', augment=False)
        args.num_classes = trainset.num_class
        args.img_size = trainset.img_size
        args.channel = 3
        # train_sampler = CategoriesSampler(trainset.label,
        #                                   args.episode_train,
        #                                   args.way_train,
        #                                   args.num_sup_train + args.num_qur_train)
        # train_loader = DataLoader(dataset=trainset,
        #                           num_workers=8,
        #                           batch_sampler=train_sampler,
        #                           pin_memory=True)
        valset = flower(setname='meta_val', augment=False)
        val_sampler = CategoriesSampler(valset.label,
                                        args.episode_test,
                                        args.way_test,
                                        args.num_sup_test + args.num_qur_test)
        val_loader = DataLoader(dataset=valset,
                                num_workers=8,
                                batch_sampler=val_sampler,
                                pin_memory=True)
        testset = flower(setname='meta_test', augment=False, noTransform=noTransform_test)
        test_sampler = CategoriesSampler(testset.label,
                                         args.episode_test,
                                         args.way_test,
                                         args.num_sup_test + args.num_qur_test)
        test_loader = DataLoader(dataset=testset,
                                 num_workers=8,
                                 batch_sampler=test_sampler,
                                 pin_memory=True)
        return None, val_loader, test_loader
    else:
        ValueError('not implemented!')
    #return None, val_loader, test_loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False

def get_model(args,type,set_maml_value=True,arbitrary_input=False):
    set_maml(set_maml_value)
    way = args.way_train
    if type == 'conv4':
        model_maml = Conv4(flatten=True, out_dim=way, img_size=args.img_size,arbitrary_input=arbitrary_input,channel=args.channel)
    elif type == 'resnet34':
        model_maml = ResNet34(flatten=True, out_dim=way)
    elif type == 'resnet18':
        model_maml = ResNet18(flatten=True, out_dim=way)
    elif type == 'resnet50':
        model_maml = ResNet50(flatten=True, out_dim=way)
    elif type=='resnet10':
        model_maml = ResNet10(flatten=True, out_dim=way)
    else:
        raise NotImplementedError
    return model_maml

bias={'cifar100':0,'miniimagenet':64,'cub':128,'flower':228,'cropdiseases':299,'eurosat':337,'isic':347,'chest':354,'omniglot':361,'mnist':425}
dataset_classnum={'cifar100':64,'miniimagenet':64,'cub':100,'flower':71,'cropdiseases':38,'eurosat':10,'isic':7,'chest':7,'omniglot':64,'mnist':10}


def set_maml(flag):
    network.ConvBlock.maml = flag
    network.SimpleBlock.maml = flag
    network.BottleneckBlock.maml = flag
    network.ResNet.maml = flag
    network.ConvNet.maml = flag

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def data2supportquery(args,data):
    way = args.way_test
    num_sup = args.num_sup_test
    num_qur = args.num_qur_test
    label = torch.arange(way, dtype=torch.int16).repeat(num_qur+num_sup)
    label = label.type(torch.LongTensor)
    label = label.cuda()
    support=data[:way*num_sup]
    support_label=label[:way*num_sup]
    query=data[way*num_sup:]
    query_label=label[way*num_sup:]
    return support,support_label,query,query_label


def label_abs2relative(specific, label_abs):
    trans = dict()
    for relative, abs in enumerate(specific):
        trans[abs] = relative
    label_relative = []
    for abs in label_abs:
        label_relative.append(trans[abs.item()])
    return torch.LongTensor(label_relative)



def pretrain(args,specific,device):
    if args.dataset=='cifar100':
        train_dataset = Cifar100_Specific(setname='meta_train', specific=specific, mode='train')
        #assert len(train_dataset)==args.way_pretrain*480, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        test_dataset = Cifar100_Specific(setname='meta_train', specific=specific, mode='test')
        #assert len(test_dataset) == args.way_pretrain*120, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,pin_memory=True)
        channel=3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='miniimagenet':
        train_dataset = MiniImageNet_Specific(setname='meta_train', specific=specific, mode='train')
        assert len(train_dataset) == args.way_pretrain*480, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = MiniImageNet_Specific(setname='meta_train', specific=specific, mode='test')
        assert len(test_dataset) == args.way_pretrain*120, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='cub':
        train_dataset = CUB_Specific(setname='meta_train', specific=specific, mode='train')
        #assert len(train_dataset) == 2400, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = CUB_Specific(setname='meta_train', specific=specific, mode='test')
        #assert len(test_dataset) == 600, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    elif args.dataset=='flower':
        train_dataset = flower_Specific(setname='meta_train', specific=specific, mode='train')
        #assert len(train_dataset) == 2400, 'error'
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                   pin_memory=True)
        test_dataset = flower_Specific(setname='meta_train', specific=specific, mode='test')
        #assert len(test_dataset) == 600, 'error'
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True, num_workers=8,
                                                  pin_memory=True)
        channel = 3
        num_epoch = 60
        learning_rate = 0.01
    else:
        raise NotImplementedError
    set_maml(False)
    if args.pre_backbone=='conv4':
        teacher=Conv4(flatten=True, out_dim=args.way_pretrain, img_size=train_dataset.img_size,arbitrary_input=False,channel=channel).cuda(device)
        optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        #optimizer=torch.optim.SGD(params=teacher.parameters(),lr=learning_rate,momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[70], gamma=0.2)#70 default#[30, 50, 80]
    elif args.pre_backbone=='resnet18':
        teacher=ResNet18(flatten=True,out_dim=args.way_pretrain).cuda(device)
        #optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        optimizer=torch.optim.SGD(params=teacher.parameters(),lr=learning_rate,momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    elif args.pre_backbone=='resnet10':
        teacher = ResNet10(flatten=True, out_dim=args.way_pretrain).cuda(device)
        # optimizer = torch.optim.Adam(params=teacher.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(params=teacher.parameters(), lr=learning_rate, momentum=.9, weight_decay=5e-4)
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[30, 50, 80], gamma=0.2)
    #train
    best_pre_model=None
    best_acc=None
    not_increase=0
    if args.fake_pretrain:
        correct, total = 0, 0
        teacher.eval()
        for batch_count, batch in enumerate(test_loader):
            image, abs_label = batch[0].cuda(device), batch[1].cuda(device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(device)
            logits = teacher(image)
            prediction = torch.max(logits, 1)[1]
            correct = correct + (prediction.cpu() == relative_label.cpu()).sum()
            total = total + len(relative_label)
        test_acc = 100 * correct / total
        return teacher.state_dict(),test_acc
    for epoch in range(num_epoch):
        # train
        teacher.train()
        for batch_count, batch in enumerate(train_loader):
            optimizer.zero_grad()
            image, abs_label = batch[0].cuda(device), batch[1].cuda(device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(device)
            logits = teacher(image)
            criteria = torch.nn.CrossEntropyLoss()
            loss = criteria(logits, relative_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher.parameters(), 50)
            optimizer.step()
        lr_schedule.step()
        correct, total = 0, 0
        teacher.eval()
        for batch_count, batch in enumerate(test_loader):
            image, abs_label = batch[0].cuda(device), batch[1].cuda(device)
            relative_label = label_abs2relative(specific=specific, label_abs=abs_label).cuda(device)
            logits = teacher(image)
            prediction = torch.max(logits, 1)[1]
            correct = correct + (prediction.cpu() == relative_label.cpu()).sum()
            total = total + len(relative_label)
        test_acc = 100 * correct / total
        if best_acc==None or best_acc<test_acc:
            best_acc=test_acc
            best_epoch=epoch
            best_pre_model=teacher.state_dict()
            not_increase=0
        else:
            not_increase=not_increase+1
            if not_increase==60:#7 for cifar and mini; 20 for omniglot
                print('early stop at:',best_epoch)
                break
        #print('epoch{}acc:'.format(epoch),test_acc,'best{}acc:'.format(best_epoch),best_acc)

    return best_pre_model,best_acc
def pretrains(args,num):
    if args.fake_pretrain == False:
        pretrained_path = args.pretrained_prefix+'/{}_{}/{}way/model'.format(args.dataset, args.pre_backbone, args.way_pretrain)
    else:
        pretrained_path = args.pretrained_prefix+'/{}_{}/{}way/model_fake'.format(args.dataset, args.pre_backbone,args.way_pretrain)
    logger = get_logger(pretrained_path, output=pretrained_path + '/' + 'log_pretrain.txt')
    timer=Timer()
    for i in range(num):
        specific=random.sample(range(dataset_classnum[args.dataset]),args.way_pretrain)
        teacher,acc=pretrain(args,specific,args.device)
        logger.info('id:{}, specific:{}, acc:{}'.format(i,specific,acc))
        torch.save({'teacher':teacher,'specific':specific,'acc':acc},os.path.join(pretrained_path,'model_specific_acc_{}.pth'.format(i)))
        print('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((i+1) / (num)))
        )


def pretrains_unique(args):
    if args.fake_pretrain == False:
        pretrained_path = args.pretrained_prefix+'/{}_{}/{}way/model_unique'.format(args.dataset, args.pre_backbone, args.way_pretrain)
    else:
        pretrained_path = args.pretrained_prefix+'/{}_{}/{}way/model_fake_unique'.format(args.dataset, args.pre_backbone,args.way_pretrain)
    logger = get_logger(pretrained_path, output=pretrained_path + '/' + 'log_pretrain.txt')
    timer=Timer()
    num=math.ceil(dataset_classnum[args.dataset]/float(args.way_train))
    for i in range(num):
        if i!=12:
            continue
        specific=list(range(i*args.way_pretrain,min(dataset_classnum[args.dataset],i*args.way_pretrain+args.way_pretrain)))
        if len(specific)!=args.way_pretrain:
            need=args.way_pretrain-len(specific)
            for n in range(need):
                specific.append(n)
        print(i,':',specific)
        teacher,acc=pretrain(args,specific,args.device)
        logger.info('id:{}, specific:{}, acc:{}'.format(i,specific,acc))
        torch.save({'teacher':teacher,'specific':specific,'acc':acc},os.path.join(pretrained_path,'model_specific_acc_{}.pth'.format(i)))
        print('ETA:{}/{}'.format(
            timer.measure(),
            timer.measure((i+1) / (num)))
        )



def one_hot(label_list,class_num):
    temp_label=label_list.reshape(len(label_list),1)
    y_one_hot = torch.zeros(size=[len(label_list), class_num],device='cuda:0').scatter_(1, temp_label, 1)
    return y_one_hot

def shuffle_task(args,support,support_label,query,query_label):
    support_label_pair=list(zip(support,support_label))
    np.random.shuffle(support_label_pair)
    support,support_label=zip(*support_label_pair)
    support=torch.stack(list(support),dim=0).cuda(args.device)
    support_label=torch.tensor(list(support_label)).cuda(args.device)

    query_label_pair = list(zip(query, query_label))
    np.random.shuffle(query_label_pair)
    query, query_label = zip(*query_label_pair)
    query = torch.stack(list(query), dim=0).cuda(args.device)
    query_label = torch.tensor(list(query_label)).cuda(args.device)

    return support,support_label,query,query_label

def construct_model_pool(args,unique=True):
    model_path_list=[]
    #ID dataset
    if unique==True:
        normal_num=math.ceil(dataset_classnum[args.dataset]/float(args.way_pretrain))
        for i in range(normal_num):
            model_path_list.append(args.pretrained_prefix+'/{}_{}/{}way/model_unique/model_specific_acc_{}.pth'.format(args.dataset, args.pre_backbone, args.way_pretrain,i))
    else:
        normal_num=args.APInum
        for i in range(normal_num):
            model_path_list.append(args.pretrained_prefix+'/{}_{}/{}way/model/model_specific_acc_{}.pth'.format(args.dataset, args.pre_backbone, args.way_pretrain,i))

    return model_path_list