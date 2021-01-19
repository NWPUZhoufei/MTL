# code is based on https://github.com/katerakelly/pytorch-maml
import torch
import random
import numpy as np
from torch.utils.data.sampler import Sampler

SEED = 3
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class BaseLoader(torch.utils.data.Dataset):
    def __init__(self, root_path='/HI_DATA/', transform=None, seed=0):
        np.random.seed(seed)
        self.transform = transform
        data_path = root_path + 'base_data.npy'
        label_path = root_path + 'base_label.npy'
        self.base_data = np.load(data_path)
        self.base_label = np.load(label_path)
        
    def __getitem__(self, index):
        image = self.base_data[index]
        label = self.base_label[index]
        if self.transform is not None:
            image = self.transform(image)
      
        return image, label

    def __len__(self):
        return len(self.base_label)

 
def get_image_index_for_class(label, class_num):
    index = []
    for i in range(class_num):
        current_class_index = []
        current_class_label = i
        for j in range(len(label)):
            if label[j] == current_class_label:
                current_class_index.append(j)
        index.append(current_class_index)
    return index

class BaseSampler(Sampler):
    def __init__(self, root_path='/HI_DATA/', num_of_class=5, num_per_class=16, n_class=55):
        label_path = root_path + 'base_label.npy'
        self.label = np.load(label_path)
        self.num_per_class = num_per_class
        self.num_of_class = num_of_class
        self.n_class = n_class
        self.index = get_image_index_for_class(self.label, self.n_class)  
    def __iter__(self):
        class_list = range(self.n_class)  
        class_list = np.random.choice(class_list, self.num_of_class, replace=False)  
        batch = []  
        for j in range(self.num_of_class):  
            current_class_index = self.index[class_list[j]]  
            current_samples_idx = np.arange(len(current_class_index))  
            current_choose_samples = np.random.choice(current_samples_idx, size=self.num_per_class, replace=False)  
            current_class_samples_index = np.array(current_class_index)[current_choose_samples]  
            batch.append(current_class_samples_index)  
        batch = [item for sublist in batch for item in sublist]

        return iter(batch)

    def __len__():
        return 1

