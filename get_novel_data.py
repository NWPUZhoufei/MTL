import pathlib
import random
import scipy.io as sio
import numpy as np

def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

class Data_process():

    def __init__(self, data_path = '/HI_DATA/', data_name = 'PaviaU', train_num = 5, seed = 103):
        self.train_num = train_num 
        self.data_path = data_path
        self.data_name = data_name
        self.cube_size = 9
        self.seed = int(seed) 
        self.fix_seed = True
        self.data_dict = sio.loadmat(str(pathlib.Path(self.data_path, self.data_name + '.mat')))
        self.data_gt_dict = sio.loadmat(str(pathlib.Path(self.data_path, self.data_name+'_gt.mat')))
        data_name = [t for t in list(self.data_dict.keys()) if not t.startswith('__')][0]
        data_gt_name = [t for t in list(self.data_gt_dict.keys()) if not t.startswith('__')][0]
        self.data = self.data_dict[data_name]
        self.data = max_min(self.data).astype(np.float32)
        self.data_gt = self.data_gt_dict[data_gt_name].astype(np.int64)
        self.dim = self.data.shape[2]
 
        if self.data_name == 'PaviaU':
            self.band_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,\
                        37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,\
                        69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,\
                       100,101,102,103]
        if self.data_name == 'Pavia':
            self.band_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,\
                         36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,60,61,62,63,64,65,66,67,\
                         68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,\
                         99,100,101,102]
        if self.data_name == 'Salinas':
            self.band_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,31,33,34,35,36,37,\
                        38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,\
                        69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,\
                        100,126,139,204]
        self.band_list = [self.band_list[i] - 1 for i in range(len(self.band_list))]
        self.data_init()
        

    def neighbor_add(self,row, col, w_size=3):
        t = w_size // 2
        cube = np.zeros(shape=[w_size, w_size, self.data.shape[2]])
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                if i + row < 0 or i + row >= self.data.shape[0] or j + col < 0 or j + col >= self.data.shape[1]:
                    cube[i + t, j + t] = self.data[row, col]  
                else:
                    cube[i + t, j + t] = self.data[i + row, j + col]
        return cube

    def data_init(self):
        
        data_gt = self.data_gt
        class_num = np.max(data_gt)
        data_pos = {i: [] for i in range(1, class_num + 1)}
        for i in range(data_gt.shape[0]):
            for j in range(data_gt.shape[1]):
                for k in range(1, class_num + 1):
                    if data_gt[i, j] == k:
                        data_pos[k].append([i, j])
        self.data_pos = data_pos  # {class_id : [[x1, y1],[x2,y2]],...}
        if self.fix_seed:
            random.seed(self.seed)
        train_pos = dict()
        test_pos = dict()
        for k, v in data_pos.items():
            if self.train_num > 0 and self.train_num < 1:
                train_num = int(self.train_num * len(v))
            else:
                train_num = self.train_num
            if len(v) < train_num:
                train_num = 15
            train_pos[k] = random.sample(v, train_num)
            test_pos[k] = [i for i in v if i not in train_pos[k]]
        self.train_pos = train_pos # labeled train_data and their location, {class_id:[loc1, loc2...], ...}
        self.test_pos = test_pos # unlabeled data and their position
        self.train_pos_all = list() # [[class_id, pos],[class_id, pos],...,]
        self.test_pos_all = list()
        for k, v in train_pos.items():
            for t in v:
                self.train_pos_all.append([k, t])
        for k, v in test_pos.items():
            for t in v:
                self.test_pos_all.append([k, t])
        self.train_data_num = len(self.train_pos_all)
        self.test_data_num = len(self.test_pos_all)
  
    def read_data(self):

        train_data_to_save = np.zeros([self.train_data_num, self.cube_size, self.cube_size, 100])
        train_label_to_save = np.zeros(self.train_data_num)

        # train_data
        for idx, i in enumerate(self.train_pos_all):
            [r,c] = i[1]
            pixel_t = self.neighbor_add(r,c,w_size=self.cube_size).astype(np.float32)
            # 在这里进行band选择
            pixel_t = pixel_t[:,:,self.band_list]
            label_t = np.array(np.array(i[0] - 1).astype(np.int64))
            train_data_to_save[idx] = pixel_t
            train_label_to_save[idx] = label_t
            #print('train_data_saved==%d' % idx)
        
        train_data_pos = []
        for i in range(len(self.train_pos_all)):
            train_data_pos.append(self.train_pos_all[i][1])

        # test data
        test_data_to_save = np.zeros([self.test_data_num, self.cube_size, self.cube_size, 100])
        test_label_to_save = np.zeros(self.test_data_num)
        
        for idx, i in enumerate(self.test_pos_all):
            [r, c] = i[1]
            pixel_t = self.neighbor_add(r,c,w_size=self.cube_size).astype(np.float32)
 
            pixel_t = pixel_t[:,:,self.band_list]
            label_t = np.array(np.array(i[0] - 1).astype(np.int64))
            test_data_to_save[idx] = pixel_t
            test_label_to_save[idx] = label_t
            #print('test_data_saved==%d'%idx)
            
        test_data_pos = []
        for i in range(len(self.test_pos_all)):
            test_data_pos.append(self.test_pos_all[i][1])

        return train_data_to_save, train_label_to_save, train_data_pos, test_data_to_save, test_label_to_save, test_data_pos
        


if __name__ == '__main__':

    data_process = Data_process(data_path = '/HI_DATA/', data_name = 'PaviaU')
    train_data_to_save, train_label_to_save, train_data_pos, test_data_to_save, test_label_to_save, test_data_pos = data_process.read_data()
    



