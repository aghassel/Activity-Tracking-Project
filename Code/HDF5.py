import numpy as np
import h5py
import pandas as pd

matrix_1 = np.random.random(size=(1000,1000))
matrix_2 = np.random.random(size=(1000,1000))
matrix_3 = np.random.random(size=(1000,1000))
#Kyle's Data Files
kJ_hand = pd.read_csv('kjh_data.csv')
kJ_front = pd.read_csv('kjf_data.csv')
kJ_back = pd.read_csv('kjb_data.csv')
kW_hand = pd.read_csv('kwh_data.csv')
kW_front = pd.read_csv('kwf_data.csv')
kW_back = pd.read_csv('kwb_data.csv')
#Abdellah's Data Files
aJ_backright = pd.read_csv('ajbrp_data.csv')
aJ_frontleft = pd.read_csv('ajflp_data.csv')
aW_backright = pd.read_csv('awbrp_data.csv')
aW_frontleft = pd.read_csv('awflp.csv')
#Liam's Data Files
lJ_backleft = pd.read_csv('ljlbp_data.csv')
lW_lefthand = pd.read_csv('lwlh_data.csv')
lW_leftpocket = pd.read_csv('lwlp_data.csv')
lW_righthand = pd.read_csv('lwrh_data.csv')
lW_rightpocket = pd.read_csv('lwrp_data.csv')

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
    K1.create_dataset('k_jump_hand', data=kJ_hand)
    K1.create_dataset('k_jump_front', data=kJ_front)
    K1.create_dataset('k_jump_back', data=kJ_back)
    K1.create_dataset('k_walk_hand', data=kW_hand)
    K1.create_dataset('k_walk_front', data=kW_front)
    K1.create_dataset('k_walk_back', data=kW_back)

    A1 = hdf.create_group('/Abdellah')
    A1.create_dataset('a_jump_backright', data=aJ_backright)
    A1.create_dataset('a_jump_frontleft', data=aJ_frontleft)
    A1.create_dataset('a_walk_backright', data=aW_backright)
    A1.create_dataset('a_walk_frontleft', data=aW_frontleft)

    L1 = hdf.create_group('/Liam')
    L1.create_dataset('l_jump_leftback', data=lJ_backleft)
    L1.create_dataset('l_walk_lefthand', data=lW_lefthand)
    L1.create_dataset('l_walk_leftpocket', data=lW_leftpocket)
    L1.create_dataset('l_walk_righthand', data=lW_lefthand)
    L1.create_dataset('l_walk_leftpocket', data=lW_leftpocket)
 


with h5py.File('hdf5_data.h5', 'r') as hdf:
    items = list(hdf.items())
    print(items)
    Matt_Group = hdf.get('/Kyle')
    print(list(Kyle.items()))
    d1 = Kyle.get('jump_hand')
    d1 = np.array(d1)
    print(d1.shape)