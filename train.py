import argparse
import os
import torch
import torchvision.transforms as transforms
import get_eposide_base_data as base_loader
import numpy as np
import random
import model as model_e
from classification_heads import ClassificationHead
import torch.nn.functional as F
from utils import set_gpu, mkdir_p, one_hot, count_accuracy

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default='/data2/zhougfei/HI_DATA/',
                    help='path to dataset')
parser.add_argument('--data_name', metavar='DIR', default='PaviaU',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--N_way', default=20, type=int,
                    metavar='NWAY', help='N_way (default: 5)')
parser.add_argument('--N_shot', default=1, type=int,
                    metavar='NSHOT', help='N_shot (default: 1)')
parser.add_argument('--test_way', default=9, type=int,
                    metavar='NWAY', help='N_way (default: 5)')
parser.add_argument('--test_shot', default=5, type=int,
                    metavar='NSHOT', help='N_shot (default: 1)')
parser.add_argument('--N_query', default=19, type=int,
                    metavar='NQUERY', help='N_query (default: 15)')
parser.add_argument('--gpu', default='0')
parser.add_argument('--eps', type=float, default=0.1, help='epsilon of label smoothing')
parser.add_argument('--pretrain_path', default='', type=str, metavar='pretrain_path',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--run_number', default='', type=int, metavar='0',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--mini_batch_size', default=1000, type=int, metavar=1000,
                    help='path to latest checkpoint (default: none)')

SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main():
    
    global args
    args = parser.parse_args()
    set_gpu(args.gpu)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    model_E = model_e.Embedding().cuda()
    model_C = ClassificationHead(base_learner='SVM-CS').cuda()
    optimizer = torch.optim.Adam([{"params":model_E.parameters()}, {"params":model_C.parameters()}], lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    
    base_dataset = base_loader.BaseLoader(
        args.data_path,
        transforms.Compose([
            transforms.ToTensor(),
        ]))

    base_set_loader = torch.utils.data.DataLoader(
        base_dataset, batch_size=args.N_way* (args.N_shot+args.N_query), shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=base_loader.BaseSampler(args.data_path, num_of_class=args.N_way, num_per_class=args.N_shot + args.N_query, n_class=55))
  
    
    train_loss, train_acc = 0, 0, 0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch > 0:
            for p in optimizer.param_groups:
                p['lr'] /= 1.11111
                p['lr'] = max(1e-6, p['lr'])
        lr = optimizer.param_groups[0]['lr']
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, lr))
        print('phase: meta_train...')
        train_loss, train_acc = train(base_set_loader, model_E, model_C, optimizer, epoch)
        print('meta_train:', epoch+1, 'loss:', train_loss, 'acc:', train_acc)
        
def train(base_set_loader, model_E, model_C, optimizer, epoch):
    model_E.train()
    model_C.train()
    accuracies = []
    losses = []
    for inter in range(1000):
        input, target = base_set_loader.__iter__().next()
        input = input.cuda()
        support_input = input.view(args.N_way, args.N_query + args.N_shot, 100, 9, 9)[:,-args.N_shot:,:,:,:].contiguous().view(-1, 100, 9, 9)
        query_input   = input.view(args.N_way, args.N_query + args.N_shot, 100, 9, 9)[:,:-args.N_shot,:,:,:].contiguous().view(-1, 100, 9, 9)
        support_input = model_E(support_input) # (way*shot,640)
        query_input = model_E(query_input)  # (way*query,640)
        #  （episodes_per_batch， num_sample, feature_size）
        support_input = support_input.unsqueeze(0) # （1，way*shot,640）
        query_input = query_input.unsqueeze(0) # （1，way*query,640）

        labels_support = np.tile(range(args.N_way), args.N_shot)
        labels_support.sort()
        labels_support = torch.cuda.LongTensor(labels_support)
        labels_support = labels_support.unsqueeze(0) # （1，way*shot）
        
        labels_query = np.tile(range(args.N_way), args.N_query)
        labels_query.sort()
        labels_query = torch.cuda.LongTensor(labels_query)
        labels_query = labels_query.unsqueeze(0) # （1，way*query）
        
        logit_query = model_C(query_input, support_input, labels_support, args.N_way, args.N_shot)
        
        smoothed_one_hot = one_hot(labels_query.reshape(-1), args.N_way)
        smoothed_one_hot = smoothed_one_hot * (1 - args.eps) + (1 - smoothed_one_hot) * args.eps / (args.N_way - 1)
    
        log_prb = F.log_softmax(logit_query.reshape(-1, args.N_way), dim=1)
        loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        loss = loss.mean()
        acc = count_accuracy(logit_query.reshape(-1, args.N_way), labels_query.reshape(-1))
        
        accuracies.append(acc.item())
        losses.append(loss.item())
        if (inter+1) % 100 == 0:
            loss_avg = np.mean(np.array(losses))
            acc_avg = np.mean(np.array(accuracies))
            #print('meta_train:', inter+1, 'loss:', loss_avg, 'acc:', acc_avg)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    loss_avg = np.mean(np.array(losses))
    acc_avg = np.mean(np.array(accuracies))       
    return loss_avg, acc_avg


       
if __name__ == '__main__':
    main()
