''' 
A script for picking out the instances on testset where a model was correct or incorrect.
''' 

import util
import available_datasets
import numpy as np
from keras.utils import to_categorical
import pandas as pd
import os
from keras import models

#Variable initialisation
agumentation=False
flavor = 'N4cor_WarpedSegmentationPosteriors2'

experiments_list =  ['261022_164054',   '271022_164133',    '281122_144817',    '251122_180422',
                    '021122_104936',    '031122_101114',    '071122_094130',    '101122_150350' ]   
model_type =        ['ResNet-ua',       'ResNet-a',         'DenseNet-ua',      'DenseNet-a',
                    'AlexNet-a',        'Alexnet-ua',       'VGG-ua',           'VGG-a']   
         
#Chosen because - highest average binary(cn vs ad/mci) test accuracy on this fold.
folds = ['cv2']               
fold = folds[0] 

#path to directory where all the experiments will be saved.
dir_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments'


# Read the test set (ADNI3).  
# This can be changed to other data-cohorts, like AIBL, EDSD etc.
dataset_marker = 'ADNI3'
dataset_obj = available_datasets.load_ADNI3_data( x_range_from = 12, x_range_to = 181, 
                                            y_range_from = 13, y_range_to = 221,
                                            z_range_from = 0, z_range_to = 179, 
                                            flavor = flavor,
                                            aug=agumentation)

images = dataset_obj['images']
groups = dataset_obj['groups']
labels = dataset_obj['labels']
labels = to_categorical(labels)
covariates = dataset_obj['covariates']


# dataframe for saving ground truth and the predictions made
df_tptn = pd.DataFrame(columns=['grps'])
grp_list = groups.Group.to_list() 
df_tptn['grps'] = grp_list     #Ground Truth: 1-CN, 2-MCI, 3-AD

#Looping over all the experiments
for exp,m_type in zip(experiments_list,model_type):
            
    experiment_path = os.path.join(dir_path, exp)
    fold_path = os.path.join(experiment_path, fold)
    
    if not os.path.exists(fold_path):
        continue

    #path where the trained model weights are saved
    model_code = 'model_{}-vgg.best.hdf5'.format(fold)
    model_path = os.path.join(fold_path, model_code)

    #Expected input and output shapes
    ip_shape = (179,169,208,1)
    op_shape = 2
        
    #intialising the models
    model = None
    if m_type == 'ResNet':
        model =  util.model_Resnet(ip_shape,op_shape)
    elif m_type == 'DenseNet':
        model =  util.DenseNet(ip_shape,op_shape)
    elif m_type == 'AlexNet':
        model =  util.model_Alex2(ip_shape,op_shape)
    else:
        model = util.model_VGG(ip_shape, op_shape)

    #loading the trained weights/parameters
    try:
        model = models.load_model(model_path)
    except:
        continue

    #predictions made by a model for a given data-cohort    
    pred = model.predict(images, batch_size=8)
    tp_tn_list = np.round(pred[:,1])==labels[:,1]  #bool array. True: either TP/TN. Flase: either FP/FN.
    df_tptn[m_type] = tp_tn_list                   #Adding predictions to the dataframe

#saving the dataframe
df_tptn.to_csv(os.path.join(dir_path, '{}_tptn.csv'.format(dataset_marker)))              