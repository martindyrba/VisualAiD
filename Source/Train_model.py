'''
A script for training a 3D-CNN model.
Possible to chose a model architecture from ResNet, DenseNet, VGG and AlexNet.     
'''

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import keras
from keras import models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

import util
import os
import json
from datetime import datetime

import logging
logging.getLogger('tensorflow').disabled=True



#Set Seeds
seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)

#Read Data
testset='adni3' #The stand alone test set (Also could be considered the left-out-set of a leave-one-side-out cv procedure).
[images, labels, groups, covariates, numfiles], [imagesT, groupsT, labelsT, covariatesT] = util.data_processing(agumentation=False)

#Weigh the classes
class_weights = util.get_class_weights(labels, numfiles)

#Set experiment name and directory
dir_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments'
dt = datetime.now().strftime('%d%m%y_%H%M%S')
experiment_path_date = os.path.join(dir_path, dt)
#os.makedirs(experiment_path, exist_ok=True)




#Model Architecture and training
acc_AD, acc_MCI, auc_AD, auc_MCI = [], [], [], []
bacc_AD, bacc_MCI = [], []
sen_AD, sen_MCI, spec_AD, spec_MCI = [], [], [], []
ppv_AD, ppv_MCI, npv_AD, npv_MCI = [], [], [], []
f1_AD, f1_MCI = [], []

# a sklearn object providing the indices to split data in train/test sets.
skf = StratifiedKFold(  n_splits=10, 
                        shuffle=True, 
                        random_state=np.random.RandomState(seed)) 

#Cross-validation loop, with 10 folds.
for k, (train_idX, validation_idX) in enumerate(skf.split(X=images, y=groups[:, 0].tolist())):  
    experiment_path = os.path.join(experiment_path_date, 'cv{}'.format(k))
    os.makedirs(experiment_path, exist_ok=True)

    # Assign trainset and valset
    traindata = images[train_idX, :]
    train_Y = labels[train_idX, :]
    
    valdata = images[validation_idX, :]
    val_Y = labels[validation_idX, :]
    val_sample_ids = groups[:,1][validation_idX]   

    #model intialisation
    #Can change the model to be trained by changeing the model call. 
    #Try: util.model_VGG, util.model_Alex2, util.model_Resnet or util.DenseNet
    model_code = 'DenseNet'
    model = util.DenseNet(ip_shape = images.shape[1:],               
                        op_shape = labels.shape[1])

    printmodelsummary = False
    if printmodelsummary:
        model.summary()

    opt = keras.optimizers.Adam(lr=0.0001)     #Used during the BVM'23 0.0001.
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)

    batch_size = 8
    epochs = 100
    filepathbest = "model_cv{}-{}.best.hdf5".format(k,model_code)   #k = cross_validation fold used
    filepathbest = os.path.join(experiment_path, filepathbest)
    
    # Fit model to training data
    #'early stopping' causes the training to stop when the monitored metric stagnates
    #while, 'model checkpoint' saves the model with the best metric values at the end of a training epoch. 
        #Therefore, no need to explcitly save the model after training.
    hist = model.fit( traindata, train_Y, 
                      batch_size=batch_size, epochs=epochs, verbose=1, 
                      shuffle=True,
                      validation_data=(valdata, val_Y),  
                      callbacks=[EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=3, min_delta=0.01),     # 0.01 = 20 eps | .001 = 60 eps    
                      ModelCheckpoint(filepathbest, monitor='loss', verbose=0, save_best_only=True, mode='auto')],       #monitor='loss',mode='auto' | 'val_acc','max' 
                      class_weight=class_weights
                    )

    #------------ post-training validation set metrics --------------------
    #Post training, saving of the metrics.
    util.save_train_plts(hist, experiment_path, True)    
    mymodel = hist.model

    print('validating model %s' % filepathbest)
    mymodel = models.load_model(filepathbest)
    pred = mymodel.predict(valdata, batch_size=batch_size)   #imagesT

    #Save validation data (Sample id, model's confidence, model's predicted labels, ground truth)
    util.save_pred(val_sample_ids, pred[:,1], np.round(pred[:,1]), val_Y[:,1], experiment_path, 'validation')  

    # Calculate accuracy/ROC-AUC for the validation/test data
    val_acc = (np.mean((np.round(pred[:,1])==val_Y[:,1]))*100)   
    print("Validation acc: %.2f%%" % val_acc)   
    f = open(os.path.join(experiment_path,'Validation_acc.txt'), "w")
    f.write(f"{val_acc}\n")
    f.close()

    util.ROC_AUC(val_Y, pred, experiment_path,'validation')
    util.binary_auc_metric(groups[validation_idX], val_Y, pred, experiment_path, marker='validation')

    #print the validation confusion matrix and save it. 
    print('confusion matrix')
    confmat_val = confusion_matrix( (groups[validation_idX][:,0]-1).tolist() , np.round(pred[:, 1]))  #groupsT['Group']-1
    print(confmat_val)
    np.savetxt( os.path.join(experiment_path,'confusion_matrix.txt'),confmat_val, fmt='%u')

    #saving the metrics for the binary classification task
    # CN-vs-MCI
    bacc, sen, spec, ppv, npv, f1, acc = util.get_values(confmat_val[(0,1),0:2]) 
    row =  {'model':model_code, 'LeftOutSet':testset , 'bacc':bacc, 'sen':sen, 'spec':spec, 'ppv':ppv, 'npv':npv, 'f1':f1, 'acc':acc }
    json.dump( row, open( os.path.join(experiment_path,'metrics_val_MCI.json'), 'w' ) )
    # CN-vs-AD
    bacc, sen, spec, ppv, npv, f1, acc = util.get_values(confmat_val[(0,2),0:2]) 
    row =  {'model':model_code, 'LeftOutSet':testset , 'bacc':bacc, 'sen':sen, 'spec':spec, 'ppv':ppv, 'npv':npv, 'f1':f1, 'acc':acc }
    json.dump( row, open( os.path.join(experiment_path,'metrics_val_AD.json'), 'w' ) )


    #------------ post-training test set metrics --------------------

    #predicting on test set    
    pred_test = mymodel.predict(imagesT, batch_size=8)
    
    #Save test data (Sample id, model's confidence, model's predicted labels, ground truth)
    util.save_pred(groupsT[:,1], pred_test[:,1], np.round(pred_test[:,1]), labelsT[:,1], experiment_path, 'test')  

    #saving test accuracy in a text file  
    test_acc = (np.mean((np.round(pred_test[:,1])==labelsT[:,1]))*100)   
    f = open(os.path.join(experiment_path,'test_acc.txt'), "w")
    f.write(f"{test_acc}\n")
    f.close()

    #Ceate and save the 'receiver operating characteristic' curve.
    util.ROC_AUC(labelsT, pred_test, experiment_path,'test')
    util.binary_auc_metric(groupsT, labelsT, pred_test, experiment_path, marker='test')
    
    # creating the confusion matrix, printing it and then saving it.
    confmat_test = confusion_matrix( (groupsT[:,0]-1).tolist() , np.round(pred_test[:, 1]))  #groupsT['Group']-1
    print(confmat_test)
    np.savetxt( os.path.join(experiment_path,'test_confusion_matrix.txt'),confmat_test, fmt='%u')

    #saving the metrics for the binary classification task
    # CN-vs-MCI
    bacc, sen, spec, ppv, npv, f1, acc = util.get_values(confmat_test[(0,1),0:2]) 
    row =  {'model':model_code, 'LeftOutSet': testset , 'bacc': bacc, 'sen': sen, 'spec':spec, 'ppv':ppv, 'npv':npv, 'f1':f1, 'acc':acc }
    json.dump( row, open( os.path.join(experiment_path,'metrics_test_MCI.json'), 'w' ) )
    # CN-vs-AD
    bacc, sen, spec, ppv, npv, f1, acc = util.get_values(confmat_test[(0,2),0:2]) 
    row =  {'model':model_code, 'LeftOutSet': testset , 'bacc': bacc, 'sen': sen, 'spec':spec, 'ppv':ppv, 'npv':npv, 'f1':f1, 'acc':acc }
    json.dump( row, open( os.path.join(experiment_path,'metrics_test_AD.json'), 'w' ) )


    #Memory management for next data reload
    del traindata, train_Y, valdata, val_Y