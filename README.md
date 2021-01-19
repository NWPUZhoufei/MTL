# MTL
Pytorch source code for the following publication:

# Requirements
- Python 3.5
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [tqdm](https://pypi.python.org/pypi/tqdm)
- torch
- torchvision
- qpth

## Preparation
Process the base dataset `base_data_process.py`

## Train
```
python train.py --N_way 10 --N_shot 1 --N_query 19 

```
## Test
```
python test.py --pretrain_path your model  --data_name PaviaU  --test_way 9  --test_shot 1  --run_number 666 

```
