'''
A script for test a trained 3D-CNN model.
Saves evaluation metrics of the group sepratability tasks.
'''

import numpy as np
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical
from keras import models
import tensorflow as tf
import util
import os

import logging
logging.getLogger('tensorflow').disabled=True

import available_datasets 
import pandas as pd



'''# only allocate the GPU RAM actually required
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

'''

def get_values(conf_matrix):
    # Given a model's confusion matrix of shape 2x2, return evaluation metrics
    assert conf_matrix.shape==(2,2)
    tn, fp, fn, tp = conf_matrix.ravel()
    sen = tp / (tp+fn)
    spec = tn / (fp+tn)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    f1 = 2 * ((ppv * sen) / (ppv + sen))
    bacc = (spec + sen) / 2
    acc = (tp+tn)/(tp+tn+fp+fn) 
    return bacc, sen, spec, ppv, npv, f1, acc


#Set Seeds
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


# Read the independent test set (ADNI3). 
# Note: this choort of data, set aside as test set, was not used during training or validation.
adni3 = available_datasets.load_ADNI3_data( x_range_from = 12, x_range_to = 181, 
                                                y_range_from = 13, y_range_to = 221,
                                                z_range_from = 0, z_range_to = 179, 
                                                flavor = 'N4cor_WarpedSegmentationPosteriors2',
                                                aug=False)

imagesT = adni3['images']
groupsT = adni3['groups']
labelsT = adni3['labels']
labelsT = to_categorical(labelsT)  #ground truth
covariatesT = adni3['covariates']

del adni3   #Memory cleanup

acc_AD, acc_MCI, auc_AD, auc_MCI = [], [], [], []
bacc_AD, bacc_MCI = [], []
sen_AD, sen_MCI, spec_AD, spec_MCI = [], [], [], []
ppv_AD, ppv_MCI, npv_AD, npv_MCI = [], [], [], []
f1_AD, f1_MCI = [], []

#Variable initialisation
dir_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments'
experiments_list =   ['261022_164054', '271022_164133',  '011122_103506',
                    '021122_104936', '031122_101114', '071122_094130', '101122_150350' ]   
model_type =  ['ResNet','ResNet','DenseNet','AlexNet','Alexnet','VGG','VGG']            
folds = ['cv0', 'cv1', 'cv2', 'cv3', 'cv4', 'cv5', 'cv6', 'cv7', 'cv8', 'cv9'] 

for exp,m_type in zip(experiments_list,model_type):

    df_val_MCI = pd.DataFrame(columns= ['Fold', 'bacc', 'sen', 'spec', 'ppv', 'npv', 'f1', 'acc'])
    df_val_AD = pd.DataFrame(columns= ['Fold', 'bacc', 'sen', 'spec', 'ppv', 'npv', 'f1', 'acc'])
    df_test_MCI = pd.DataFrame(columns= ['Fold', 'bacc', 'sen', 'spec', 'ppv', 'npv', 'f1', 'acc'])
    df_test_AD = pd.DataFrame(columns= ['Fold', 'bacc', 'sen', 'spec', 'ppv', 'npv', 'f1', 'acc'])

    for fold in folds:
                
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

        #predicting on test set    
        pred = model.predict(imagesT, batch_size=8)
        test_acc = (np.mean((np.round(pred[:,1])==labelsT[:,1]))*100)   

        #saving test accuracy in a text file  
        f = open(os.path.join(fold_path,'Test_acc.txt'), "w")
        f.write(f"{test_acc}\n")
        f.close()

        #Ceate and save the 'receiver operating characteristic' curve.
        util.ROC_AUC(labelsT, pred, fold_path,'Test')

        # creating the confusion matrix, printing it and then saving it.
        confmat = confusion_matrix( (groupsT['Group']-1).tolist() , np.round(pred[:, 1]))  #groupsT['Group']-1
        print(confmat)
        np.savetxt( os.path.join(fold_path,'Test_confusion_matrix.txt'),confmat, fmt='%u')



        # binary comparison: AD vs. HC and MCI vs. HC
        #reads confusion_matrix created while training, on the validation set
        f = open(os.path.join(fold_path, 'confusion_matrix.txt'), "r")  
        confmat_val =  np.loadtxt(f)
        bacc, sen, spec, ppv, npv, f1, acc = get_values(confmat_val[(0,1),0:2]) # MCI
        row =  {'Fold': fold , 'bacc': bacc, 'sen': sen, 'spec':spec, 'ppv':ppv, 'npv':npv, 'f1':f1, 'acc':acc }
        df_val_MCI = df_val_MCI.append(row, ignore_index=True)
        bacc, sen, spec, ppv, npv, f1, acc = get_values(confmat_val[(0,2),0:2]) # AD
        row =  {'Fold': fold , 'bacc': bacc, 'sen': sen, 'spec':spec, 'ppv':ppv, 'npv':npv, 'f1':f1, 'acc':acc }
        df_val_AD = df_val_AD.append(row, ignore_index=True)

        #reads confusion_matrix created while testing
        f = open(os.path.join(fold_path, 'Test_confusion_matrix.txt'), "r")
        confmat_test =  np.loadtxt(f)
        bacc, sen, spec, ppv, npv, f1, acc = get_values(confmat_test[(0,1),0:2]) # MCI
        row =  {'Fold': fold , 'bacc': bacc, 'sen': sen, 'spec':spec, 'ppv':ppv, 'npv':npv, 'f1':f1, 'acc':acc }
        df_test_MCI = df_test_MCI.append(row, ignore_index=True)
        bacc, sen, spec, ppv, npv, f1, acc = get_values(confmat_test[(0,2),0:2]) # AD
        row =  {'Fold': fold , 'bacc': bacc, 'sen': sen, 'spec':spec, 'ppv':ppv, 'npv':npv, 'f1':f1, 'acc':acc }
        df_test_AD = df_test_AD.append(row, ignore_index=True)

    # For a given experiment, saves validation and test eval metrics, across all the folds. 
    # for each binary group sepration tasks of AD-vs-CN and MCI-vs-CN 
    df_val_MCI.to_csv(os.path.join(experiment_path, 'df_val_MCI.csv'))
    df_val_AD.to_csv(os.path.join(experiment_path, 'df_val_AD.csv')) 
    df_test_MCI.to_csv(os.path.join(experiment_path, 'df_test_MCI.csv')) 
    df_test_AD.to_csv(os.path.join(experiment_path, 'df_test_AD.csv'))    

print()