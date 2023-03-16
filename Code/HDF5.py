import numpy as np
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
import random 

#Kyle's Data Files
kJ_hand = pd.read_csv('data/jumping/kjh_data.csv')
kJ_front = pd.read_csv('data/jumping/kjf_data.csv')
kJ_back = pd.read_csv('data/jumping/kjb_data.csv')
kW_hand = pd.read_csv('data/walking/kwh_data.csv')
kW_front = pd.read_csv('data/walking/kwf_data.csv')
kW_back = pd.read_csv('data/walking/kwb_data.csv')
#Kyle's Combined Dataset 
kyle_combined = pd.concat([kJ_hand,kJ_front,kJ_back,kW_hand,kW_front,kW_back]); 
kyle_combined.to_csv("data/combined/kyle_combined_data.csv", index=False)

#Abdellah's Data Files
aJ_backright = pd.read_csv('data/jumping/ajbrp_data.csv')
aJ_frontleft = pd.read_csv('data/jumping/ajflp_data.csv')
aW_backright = pd.read_csv('data/walking/awbrp_data.csv')
aW_frontleft = pd.read_csv('data/walking/awflp_data.csv')
#Abdellah's combined dataset 
abdellah_combined = pd.concat([aJ_backright,aJ_frontleft,aW_backright,aW_frontleft]); 
abdellah_combined.to_csv("data/combined/abdellah_combined_data.csv", index=False)

#Liam's Data Files
lJ_backleft = pd.read_csv('data/jumping/ljlbp_data.csv')
lJ_backright = pd.read_csv('data/jumping/ljbrp_data.csv')
lW_lefthand = pd.read_csv('data/walking/lwlh_data.csv')
lW_leftpocket = pd.read_csv('data/walking/lwlp_data.csv')
lW_righthand = pd.read_csv('data/walking/lwrh_data.csv')
lW_rightpocket = pd.read_csv('data/walking/lwrp_data.csv')
#Liam's combined dataset 
liam_combined = pd.concat([lJ_backleft,lJ_backright,lW_lefthand,lW_leftpocket,lW_righthand,lW_rightpocket]); 
liam_combined.to_csv("data/combined/liam_combined_data.csv", index=False)

#Concatenated Dataset 
combined_dataset = pd.concat([kJ_hand,kJ_front,kJ_back,kW_hand,kW_front,kW_back,aJ_backright,aJ_frontleft,aW_backright,aW_frontleft,lJ_backleft,lJ_backright,lW_lefthand,lW_leftpocket,lW_righthand,lW_rightpocket])

#Write combined dataset to a new CSV file
combined_dataset.to_csv("data/combined/Combined_dataset.csv", index=False)



with h5py.File('hdf5_data.h5', 'w') as hdf:
    #Creating main dataset
    dataset = hdf.create_group('/dataset')
    tee = dataset.create_dataset('dataset', data=combined_dataset)

    #read in Combined_dataset.csv
    cds = pd.read_csv("data/combined/Combined_dataset.csv")
    
    group_size = 500
    groups = [cds.iloc[i:i+group_size] for i in range(0, len(cds), group_size)]
    count_groups = int(np.ceil(len(cds) / group_size))
    random.shuffle(groups)
    

    # # Shuffle group elements
    # for i in range(count_groups):
    #     groups[i] = groups[i].sample(frac=1).reset_index(drop=True)

    print(groups)

    train_data, test_data = train_test_split(cds, test_size=0.1)
       
    # Print the number of rows in each set
    # print(f"Number of rows in training set: {len(train_data)}")
    # print(f"Number of rows in testing set: {len(test_data)}")

    train_data.to_csv("data/combined/Train_Data.csv")
    test_data.to_csv("data/combined/Test_Data.csv")    

    #Create training dataset
    training_dataset = hdf.create_group('dataset/training')
    training_dataset.create_dataset('training_dataset', data=train_data)

    #Create testing dataset
    testing_dataset = hdf.create_group('dataset/testing')
    testing_dataset.create_dataset('testing_dataset', data=test_data)

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
    L1.create_dataset('l_jump_rightback', data=lJ_backright)
    L1.create_dataset('l_walk_lefthand', data=lW_lefthand)
    L1.create_dataset('l_walk_leftpocket', data=lW_leftpocket)
    L1.create_dataset('l_walk_righthand', data=lW_lefthand)
    L1.create_dataset('l_walk_rightpocket', data=lW_rightpocket)
 
# Output test
    # with h5py.File('hdf5_data.h5', 'r') as hdf:
    #     items = list(hdf.items())
    #     print(items)
    #     print(list(testing_dataset.items()))
    #     d1 = combined_dataset.get('testing_dataset')
    #     d1 = np.array(d1)
    #     print('\n')
    #     print(d1.shape)