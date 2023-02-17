'''
A script for creating entropy maps and standard deviation maps, of the relevances (method: LRP-CMPalpha1beta0) for the trained models. 
Only using CNN models trained without data agumentation. And only reporting on the standalone test set (ADNI3).  
'''

import available_datasets
from keras.utils import to_categorical
import os
import rmap_util
import numpy as np
import pandas as pd
import h5py
from entropy_estimators import continuous


#Initializations
flavor = 'N4cor_WarpedSegmentationPosteriors2'
datasets = ['ADNI2', 'AIBL', 'EDSD', 'DELCODE', 'ADNI3'] 
dataset_marker = datasets[4]     #Only reporting on the standalone test set (ADNI3).


# Only using CNN models trained without data agumentation.
experiments_list =  ['261022_164054',   '281122_144817',  '031122_101114',    '071122_094130']   
model_type =        ['ResNet-ua',       'DenseNet-ua',    'Alexnet-ua',       'VGG-ua']


# Only using the LRP-CMPalpha1beta0 method as it works better than other relevance propogation methods.
methods = [ ("lrp.sequential_preset_a", {"neuron_selection_mode": "index", "epsilon": 1e-10}, "LRP-CMPalpha1beta0")] 

#chosing fold with best aggreageted performance by the models.
folds=['cv2']

q = 0.9999

# Paths where things were stored before or where we want to store processed data. 
dir_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments'
dir_path2 = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments/mean_map'
dir_path3 = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments/entropy_map'
dir3 = os.path.join(dir_path3, dataset_marker)
os.makedirs(dir3, exist_ok=True)
data_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/data/'
im_path = data_path + 'ADNI_t1linear/AD/AD_4910_N4cor_WarpedSegmentationPosteriors2.nii.gz'


##---- function definations ---------

def aggregate_activations(flavor, agumentation, experiments_list, model_type, methods, folds , dir_path, q, dir3):
    '''
    Collecting and exporting all the activations/relevances for each model, into a binary file.

    flavor:             a segementation marker chosen after preprocessing the MRI scans. By default gray matter segementation are chosen.  
    agumentation:       a flag to decide the loaded dataset should contain agumented brain scans. Should be kept False unless training. 
    experiments_list:   list of relevant experiments where 3d-cnn models were trained. Each experiment in str(datetime) format.
    model_type:         list of unique markers for each relevant experiments, to identify the experiments in human readable manner.
    methods:            list of relevance propogation methods. Each element is a tuple like (innvestigate.RelevanceMethod, required arguments, user-defined method identifiers)
    folds:              list of cross-validation folds chosen.
    dir_path:           a directory path where all the experiments (trained model weights) are stored. 
    q:                  quantile chosen to scale the data.
    dir3:               a directory path where the aggregated activations will be exported to.
    '''
 
# Loading the test dataset.
    dataset_obj = available_datasets.load_ADNI3_data(  x_range_from = 12, x_range_to = 181, 
                                                    y_range_from = 13, y_range_to = 221, 
                                                    z_range_from = 0, z_range_to = 179, 
                                                    flavor = flavor, aug=agumentation)
        
    images = dataset_obj['images']
    groups = dataset_obj['groups']
    labels = dataset_obj['labels']
    labels = to_categorical(labels)
    covariates = dataset_obj['covariates']

    # Creating dictionary of the LRP relevance analysers, given each trained CNN model.
    exp_dict = rmap_util.create_anaylzer_dict(experiments_list, model_type, methods, folds , dir_path)


    # Looping through all CNN models and images in test set.
    for m_type in model_type:
        a_all = np.zeros((1,179,169,208))     #intialization

        for index in range(len(images)):  
            im = images[index]     
            im = np.expand_dims(im,0)  
            try:
                a = exp_dict[m_type]['LRP-CMPalpha1beta0'].analyze(im,neuron_selection=1)       #activation/relevance created by a model for an image 
                a = rmap_util.scale_relevance_map(r_map=a, clipping_threshold=1, quantile=q)    #sacling and clipping of the activations
                a = np.squeeze(a, axis=4)                                                       #dimention mangement: resulting shape (1,179,169,208)
                if index==0:
                    a_all = a
                else:
                    a_all = np.concatenate((a_all,a))                                           #concatenate activations on the 1st dimention.
            
            except:  
                #print model and image index, for failed calls to relevance proporation method.
                #print(m_type)          
                #print(index)
                continue    
        
        #saving the concatenated activations (ndarray) as h5py file
        f_name = str(os.path.join(dir3,'activations_{}.hdf5'.format(m_type)))    
        hf = h5py.File(f_name, 'w')
        hf.create_dataset('ADNI3_activations', data=a_all, compression='gzip')
        hf.close()            
        del a_all       #memory freed up



def cont_entropy_map(vector):
    '''
    Calculates (per voxel) entropy.
    NOTE:   Since the activations will be real number between the range of [-1,1], from an **unkown distribution**.
            To find distribution/prob. of activations, one could in theory try to 'bin' activations and 'count' them, 
            but this would introduce further heuristic choices in the entropy estimation process.
      
           I looked up other options: 
                    How to find entropy of Continuous variable in Python? 
                    (Link: https://stackoverflow.com/questions/45591968/how-to-find-entropy-of-continuous-variable-in-python)  
                    (Git: https://github.com/paulbrodersen/entropy_estimators)
                    (Paper: https://arxiv.org/pdf/1506.06501.pdf) 

    vector: an array containing the aggregated activations.
    '''

    #intialisation
    entropy_map = np.zeros((179,169,208))

    for i in range(vector.shape[1]):
        for j in range(vector.shape[2]):
            for k in range(vector.shape[3]):
                entropy_map[i,j,k] = continuous.get_h(vector[:,i,j,k] , k=5)    #entropy calculated per voxel

    return entropy_map

def std_dev_map(vector):
    '''
    Creates a standard deviation map, over activations. Does a voxel-based analysis.

    vector: an array containing the aggregated activations.
    '''
    
    #intialisation
    stddev_map = np.zeros((179,169,208))

    for i in range(vector.shape[1]):
        for j in range(vector.shape[2]):
            for k in range(vector.shape[3]):
                stddev_map[i,j,k] = np.std(vector[:,i,j,k] , dtype=np.float64)   #std.dev. calculated per voxel

    return stddev_map

def create_divergence_maps(dir_path2, dataset_marker, model_type, dir3, im_path, divergence):
    '''
    a wrapper function to create divergence - either entropy or std.dev., maps.
    NOTE: only creates divergence maps for the MCI class. 
    Similar logic can be build for other disease-classes, in case the need arises. 

    dir_path2:      a directory path where the model's predictions are saved.
    dataset_marker: a marker, naming a data-cohort 
    model_type:     list of unique markers for each relevant experiments, to identify the experiments in human readable manner.
    dir3:           a directory path where the divergence maps will be saved.
    im_path:        path to a sample MRI scan, to be used as header while saving any volume.
    divergence:     method to be used to create the divergence maps. In ['entropy','std']. 
    '''
    # Reading csv storing model's predictions on the testset (ADNI3). 
    df_tptn = pd.read_csv( os.path.join(dir_path2, '{}_tptn.csv'.format(dataset_marker)))

    #Looping over all the experiments
    for m_type in model_type:
        #initialisation
        a_mci = np.zeros((1,179,169,208))
        insert_flag = True
        
        #reading aggregated activation file, for a given model.
        f_path = str(os.path.join(dir3, 'activations_{}.hdf5'.format(m_type)))
        hf = h5py.File(f_path, 'r')
        hf.keys # read keys
        a_all = np.array(hf.get('ADNI3_activations'))
        hf.close()
        
        #Slicing the activation array to pick relevant instances.
        #looping over all the activations
        for index in range(len(df_tptn)):
            
            #only considering cases where model makes a correct prediction 
            if df_tptn[m_type][index]:  #TP/TN 
                
                if df_tptn['grps'][index] == 1:      #CN
                    continue
                
                #picking activations only of MCI disease classes. (Based on previous analysis)   
                elif df_tptn['grps'][index] == 2:    #MCI
                    if insert_flag:
                        a_mci = np.expand_dims(a_all[index],0)
                        insert_flag = False   #There has been the first insertion in the array, now new elements needs to be appended.
                    else:
                        a_mci = np.concatenate( (a_mci, np.expand_dims(a_all[index],0)) )                
                
                else:                                #AD
                    continue      
            
            
            #not considering the cases where model makes an incorrect prediction
            else: #FP/FN
                continue
        
        
        if divergence=='entropy':
            #creates and saves, voxel-level entropy maps.
            mci_entropy = cont_entropy_map(a_mci)
            rmap_util.activation2nifti(mci_entropy, im_path, '','',m_type,'MCI-entropy_map',dir3)

        elif divergence=='std':
            #creates and saves, voxel-level standard deviation maps.
            mci_stddev = std_dev_map(a_mci)
            rmap_util.activation2nifti(mci_stddev, im_path, '','',m_type,'MCI-std_dev_map',dir3)


##---- aggreagtion and map creation function calls -------

#Unmute this function call, when first running this script to collect all the activations.
#aggregate_activations(flavor, False, experiments_list, model_type, methods, folds , dir_path, q, dir3)

#creates and saves divergence maps
create_divergence_maps(dir_path2, dataset_marker, model_type, dir3, im_path, divergence='std')