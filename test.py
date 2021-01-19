import argparse
import os
import torch
import numpy as np
import random
import model as model_e
from classification_heads import ClassificationHead
from utils import set_gpu, mkdir_p
import get_novel_data as novle_data_loader

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_path', metavar='DIR', default='./HI_DATA/',
                    help='path to dataset')
parser.add_argument('--data_name', metavar='DIR', default='PaviaU',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
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
parser.add_argument('--test_way', default=9, type=int,
                    metavar='NWAY', help='N_way (default: 5)')
parser.add_argument('--test_shot', default=5, type=int,
                    metavar='NSHOT', help='N_shot (default: 1)')
parser.add_argument('--gpu', default='0')
parser.add_argument('--eps', type=float, default=0.1, help='epsilon of label smoothing')
parser.add_argument('--pretrain_path', default='', type=str, metavar='pretrain_path',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--run_number', default=10, type=int, metavar='0',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--mini_batch_size', default=1000, type=int, metavar='1000',
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
    pretrain_checkpoint = torch.load(args.pretrain_path)
    model_E.load_state_dict(pretrain_checkpoint['state_dict_E'])
    
    final_result_all = []
    for i in range(10):
        data_process = novle_data_loader.Data_process(data_path = args.data_path, data_name = args.data_name, train_num = args.test_shot, seed = args.run_number + i)
        train_data_to_save, train_label_to_save, train_data_pos, test_data_to_save, test_label_to_save, test_data_pos = data_process.read_data()
        query_pred = test_minibatch(train_data_to_save, train_label_to_save, test_data_to_save, model_E, model_C) # （way*query,way）
        current_result = get_performance(test_label_to_save, query_pred)
        final_result_all.append(current_result)
    final_result_all = np.array(final_result_all)
    final_result_mean = np.mean(final_result_all,0)
    final_result_std = np.std(final_result_all,0)
    print('final_result_mean:', final_result_mean)
    print('final_result_std:', final_result_std)
   
def test_minibatch(support_set, support_label, query_set, model_E, model_C):
    model_E.eval()
    model_C.eval()
    support_set = torch.cuda.FloatTensor(support_set) # (way*shot, 9,9,100)
    support_set = support_set.permute([0, 3, 1, 2])  # (way*shot, 100,9,9)
    support_set = model_E(support_set) # (way*shot,640)
    support_set = support_set.unsqueeze(0) # （1，way*shot,1024）
    support_label = torch.cuda.LongTensor(support_label) # (way*shot)
    support_label = support_label.unsqueeze(0) # （1，way*shot）   
    query_set = torch.cuda.FloatTensor(query_set) # （way*query,9,9,100）
    query_set = query_set.permute([0, 3, 1, 2]) # (way*query, 100,9,9)
    test_num = query_set.shape[0]
    mini_batch = test_num // args.mini_batch_size + 1
    query_batch = torch.rand(mini_batch*args.mini_batch_size, 100, 9, 9).cuda()
    query_batch[:test_num] = query_set
    query_batch = query_batch.view(mini_batch, args.mini_batch_size, 100, 9, 9)
    query_pred = []
    for i in range(mini_batch):
        current_query_set = query_batch[i] # (mini_batch_size,100,9,9)
        current_query_set = model_E(current_query_set)  # (mini_batch_size,1024)
        current_query_set = current_query_set.unsqueeze(0) # （1，mini_batch_size,1024)
        current_query_pred = model_C(current_query_set, support_set, support_label, args.test_way, args.test_shot) # (1,mini_batch_size,way)
        current_query_pred = current_query_pred.detach().cpu().numpy() # (1,mini_batch_size,way)
        current_query_pred = np.squeeze(current_query_pred) # (mini_batch_size, way)
        query_pred.append(current_query_pred)
    query_pred = np.array(query_pred) # (mini_batch, mini_batch_size, way)
    query_pred = query_pred.reshape(mini_batch*args.mini_batch_size, -1) # (mini_batch*mini_batch_size, way)

    query_pred = query_pred[:test_num, :] # (test_num, way)
    
    return query_pred



 
def get_performance(query_label, query_pred):
    label = query_label.astype(np.int64) # (way*query)
    pre_labels = query_pred # (way*query,way)
    pre_labels = np.argmax(pre_labels, 1).astype(np.int64)  # (way*query)
    confusion_matrix = np.zeros([args.test_way, args.test_way])
    for j in range(len(label)):
        confusion_matrix[pre_labels[j], label[j]] += 1
    matrix = confusion_matrix
    ac_list = []
    for i in range(len(matrix)):
        ac = matrix[i, i] / sum(matrix[:, i])
        ac_list.append(ac)
        print(i + 1, 'class:', '(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
    print('confusion matrix:')
    print(np.int_(matrix))
    print('total right num:', np.sum(np.trace(matrix)))
    accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
    kk = 0
    for i in range(matrix.shape[0]):
        kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
    pe = kk / (np.sum(matrix) * np.sum(matrix))
    pa = np.trace(matrix) / np.sum(matrix)
    kappa = (pa - pe) / (1 - pe)
    ac_list1 = np.asarray(ac_list)
    aa = np.mean(ac_list1)
    oa = accuracy
    final_result = [oa, aa, kappa] + ac_list
    print('###'*10)
    print(final_result)
    print('###'*10)
    final_result = np.array(final_result)
    return final_result
    
        
  
       
if __name__ == '__main__':
    main()
