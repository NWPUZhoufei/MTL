import pathlib
import scipy.io as sio
import numpy as np

def read_raw_data(data_path, data_name):
    data_dict = sio.loadmat(str(pathlib.Path(data_path, data_name + '.mat')))
    data_gt_dict = sio.loadmat(str(pathlib.Path(data_path, data_name+'_gt.mat')))
    data_name = [t for t in list(data_dict.keys()) if not t.startswith('__')][0]
    data_gt_name = [t for t in list(data_gt_dict.keys()) if not t.startswith('__')][0]
    data = data_dict[data_name]
    data = max_min(data).astype(np.float32)
    data_gt = data_gt_dict[data_gt_name].astype(np.int64)

    return data, data_gt


def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))



def get_position(data_gt):
    class_num = np.max(data_gt)
    data_pos = {i: [] for i in range(1, class_num + 1)}
    for i in range(data_gt.shape[0]):
        for j in range(data_gt.shape[1]):
            for k in range(1, class_num + 1):
                if data_gt[i, j] == k:
                    data_pos[k].append([i, j])
    return data_pos


def get_cube(data, row, col, cube_size):
    t = cube_size // 2
    cube = np.zeros(shape=[cube_size, cube_size, data.shape[2]])
    for i in range(-t, t + 1):
        for j in range(-t, t + 1):
            if i + row < 0 or i + row >= data.shape[0] or j + col < 0 or j + col >= data.shape[1]:
                cube[i + t, j + t] = data[row, col] 
            else:
                cube[i + t, j + t] = data[i + row, j + col]
    return cube


def process_data(data, data_pos, threshold, cube_size, band_list):
    select_class = []
    for key in data_pos:
        if len(data_pos[key]) > threshold:
            current_class = []
            for i in range(len(data_pos[key])):
                [r,c] = data_pos[key][i]
                current_sample = get_cube(data, r, c, cube_size).astype(np.float32)
                current_class.append(current_sample)
            current_class = np.array(current_class)
            current_class = current_class[:,:,:,band_list]
            select_class.append(current_class)
    return select_class
    
if __name__ == '__main__':

    Houston_band_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,\
                        36,37,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,\
                        68,69,70,71,72,77,107,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,\
                        126,127,128,129,130,132,133,134,135,143,144]
    Houston_band_list = [Houston_band_list[i] - 1 for i in range(len(Houston_band_list))]
    
    Botswana_band_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,\
                         36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63,64,65,66,67,\
                         68,69,70,71,72,73,88,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,\
                         127,128,137,138,139,140,141,142,143,144,145]
    Botswana_band_list = [Botswana_band_list[i] - 1 for i in range(len(Botswana_band_list))]
    KSC_band_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,31,32,33,35,36,37,39,\
           40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,71,72,\
           73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,95,101,120,132,143,144,145,146,147,\
           148,149,150,151,155,167,175,176]
    KSC_band_list = [KSC_band_list[i] - 1 for i in range(len(KSC_band_list))]
    Chikusei_band_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,\
                35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,65,66,\
                67,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,116,117,\
                118,119,120,121,122,123,124,125,126,127,128]
    Chikusei_band_list = [Chikusei_band_list[i] - 1 for i in range(len(Chikusei_band_list))]
    
    cube_size = 9
    threshold = 200
    
    data_path = 'Houston'
    data_name = 'Houston'
    data, data_gt = read_raw_data(data_path, data_name)
    data_pos = get_position(data_gt)
    Houston_select_class = process_data(data, data_pos, threshold, cube_size, Houston_band_list)
    print(len(Houston_select_class))
    
    data_path = 'Botswana'
    data_name = 'Botswana'
    data, data_gt = read_raw_data(data_path, data_name)
    data_pos = get_position(data_gt)
    Botswana_select_class = process_data(data, data_pos, threshold, cube_size, Botswana_band_list)
    print(len(Botswana_select_class))
    
    data_path = 'KSC'
    data_name = 'KSC'
    data, data_gt = read_raw_data(data_path, data_name)
    data_pos = get_position(data_gt)
    KSC_select_class = process_data(data, data_pos, threshold, cube_size, KSC_band_list)
    print(len(KSC_select_class))
    
    data = np.load("Chikusei/Chikusei.npy")
    data = max_min(data).astype(np.float32)
    data_gt = np.load("Chikusei/Chikusei_gt.npy")
    data_pos = get_position(data_gt)
    Chikusei_select_class = process_data(data, data_pos, threshold, cube_size, Chikusei_band_list)
    print(len(Chikusei_select_class))
    
    base_data_set = Houston_select_class + Botswana_select_class + KSC_select_class + Chikusei_select_class
    print(len(base_data_set))
    base_data = []
    base_label = []
    for i in range(len(base_data_set)):
        for j in range(len(base_data_set[i])):
            base_data.append(base_data_set[i][j])
            base_label.append(i)
    base_data = np.array(base_data)
    base_label = np.array(base_label) 
    base_data = np.save("base_data.npy", base_data)
    base_label = np.save("base_label.npy", base_label)

  

    
    
    
    
    

