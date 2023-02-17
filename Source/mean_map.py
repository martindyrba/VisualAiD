''' 
A script for creating the mean relevance maps, for each data-cohorts seprated by disease stages.
Relevance from models only considered when they make correct predictions.
''' 

import available_datasets
from keras.utils import to_categorical
import pandas as pd
import os
import rmap_util
import numpy as np

#Variable initialisation
agumentation=False
flavor = 'N4cor_WarpedSegmentationPosteriors2'
datasets = ['ADNI2', 'AIBL', 'EDSD', 'DELCODE', 'ADNI3'] 
dataset_marker = datasets[4]    #Picking the test set


experiments_list =  ['261022_164054',   '271022_164133',    '281122_144817',    '251122_180422',
                    '021122_104936',    '031122_101114',    '071122_094130',    '101122_150350' ]   
model_type =        ['ResNet-ua',       'ResNet-a',         'DenseNet-ua',      'DenseNet-a',
                    'AlexNet-a',        'Alexnet-ua',       'VGG-ua',           'VGG-a']

#chosen method for relevance propagation
methods = [ ("lrp.sequential_preset_a", {"neuron_selection_mode": "index", "epsilon": 1e-10}, "LRP-CMPalpha1beta0")] 
#Chosen because - highest average binary(cn vs ad/mci) test accuracy on this fold.
folds=['cv2']    
#Chosen quantile value used for relevance scaling.
q = 0.9999


#path initialisations
dir_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments'
dir_path2 = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments/mean_map'
data_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/data/'
im_path = data_path + 'ADNI_t1linear/AD/AD_4910_N4cor_WarpedSegmentationPosteriors2.nii.gz'

#Read the test set (ADNI3).
dataset_obj = available_datasets.load_ADNI3_data(  x_range_from = 12, x_range_to = 181, 
                                                y_range_from = 13, y_range_to = 221, 
                                                z_range_from = 0, z_range_to = 179, 
                                                flavor = flavor, aug=agumentation)
    
images = dataset_obj['images']
groups = dataset_obj['groups']
labels = dataset_obj['labels']
labels = to_categorical(labels)
covariates = dataset_obj['covariates']

#reading the predictions made by the trained models on a given data-cohort.
df_tptn = pd.read_csv( os.path.join(dir_path2, '{}_tptn.csv'.format(dataset_marker)))

#Creating dictionary of the LRP relevance analysers, given each trained CNN model.
exp_dict = rmap_util.create_anaylzer_dict(experiments_list, model_type, methods, folds , dir_path)


#Looping over all the experiments
for m_type in model_type:
    #variables to sum up all the relevances/activations into. 
    #specific for each disease class.
    a_cn = np.zeros((179,169,208))
    a_mci = np.zeros((179,169,208))
    a_ad = np.zeros((179,169,208))
    #counters for the number of instances in each disease class. 
    count_cn = 0
    count_mci = 0
    count_ad = 0

    #looping over all the images
    for index in range(len(images)):
        #only considering cases where model makes a correct prediction 
        if df_tptn[m_type][index]:  #TP/TN 
            im = images[index]     
            im = np.expand_dims(im,0)  
            
            #getting activation maps.
            try:
                a = exp_dict[m_type]['LRP-CMPalpha1beta0'].analyze(im,neuron_selection=1)
                a = rmap_util.scale_relevance_map(r_map=a, clipping_threshold=1, quantile=q)
                a = np.squeeze(a)
            except:
                continue

            #adding activations to respective disease classes.    
            if df_tptn['grps'][index] == 1:      #CN
                a_cn = a_cn + a  
                count_cn = count_cn +1 
            elif df_tptn['grps'][index] == 2:    #MCI
                a_mci = a_mci + a
                count_mci = count_mci +1 
            else:                                #AD      
                a_ad = a_ad + a
                count_ad = count_ad +1 
        
        #not considering the cases where model makes an incorrect prediction
        else: #FP/FN
            continue
        
    dir3 = os.path.join(dir_path2, dataset_marker+'_')
    os.makedirs(dir3, exist_ok=True)

    #saving the mean relevance maps.
    if count_cn:
        rmap_util.activation2nifti(a_cn/count_cn,   im_path, '', dataset_marker, m_type, 'CN', dir3)    #mean=sum/N 
        print('{} #CN:  {}'.format(m_type,count_cn))
    if count_mci:
        rmap_util.activation2nifti(a_mci/count_mci, im_path, '', dataset_marker, m_type, 'MCI', dir3)
        print('{} #MCI: {}'.format(m_type,count_mci))
    if count_ad:
        rmap_util.activation2nifti(a_ad/count_ad,   im_path, '', dataset_marker, m_type, 'AD', dir3)  
        print('{} #AD:  {}'.format(m_type,count_ad))    