import numpy as np
import h5py
import pandas as pd

matrix_1 = np.random.random(size=(1000,1000))
matrix_2 = np.random.random(size=(1000,1000))
matrix_3 = np.random.random(size=(1000,1000))
#Kyle's Data Files
kjh_data = pd.read_csv('kjh_data.csv')
kjf_data = pd.read_csv('kjf_data.csv')
kjb_data = pd.read_csv('kjb_data.csv')
krh_data = pd.read_csv('krh_data.csv')
krf_data = pd.read_csv('krf_data.csv')
krb_data = pd.read_csv('krb_data.csv')
kwh_data = pd.read_csv('kwh_data.csv')
kwf_data = pd.read_csv('kwf_data.csv')
kwb_data = pd.read_csv('kwb_data.csv')
# #Abdellah's Data Files

# #Liam's Data Files


with h5py.File('hdf5_data.h5', 'w') as hdf:
    #Creating main dataset
    dataset = hdf.create_group('/dataset')
    dataset.create_dataset('dataset', data=matrix_1)

    train = hdf.create_group('/dataset/train')
    train.create_dataset('train', data=matrix_2)

    test = hdf.create_group('/dataset/test')
    test.create_dataset('test', data=matrix_3)


    #Create member subgroups
    K1 = hdf.create_group('/Kyle')
    K1.create_dataset('jump_hand', data=kjh_data)
    K1.create_dataset('jump_front', data=kjf_data)
    K1.create_dataset('jump_back', data=kjb_data)
    K1.create_dataset('run_hand', data=krh_data)
    K1.create_dataset('run_front', data=krf_data)
    K1.create_dataset('run_back', data=krb_data)
    K1.create_dataset('walk_front', data=kwf_data)
    K1.create_dataset('walk_back', data=kwb_data)

    # A1 = hdf.create_group('/Abdellah')
    # A1.create_dataset('jump_hand', data=)
   

    # L1 = hdf.create_group('/Liam')
    # L1.create_dataset('jump_hand', data=)
 


with h5py.File('hdf5_data.h5', 'r') as hdf:
    items = list(hdf.items())
    print(items)
    Matt_Group = hdf.get('/Kyle')
    print(list(Kyle.items()))
    d1 = Kyle.get('jump_hand')
    d1 = np.array(d1)
    print(d1.shape)