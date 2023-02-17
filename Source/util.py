'''
A utility script, with different supporting functionalities.
'''

import available_datasets 
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc
import keras
from keras import layers
from keras.layers import Input, Conv3D, BatchNormalization, Dense
from keras.layers import AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D
from keras.models import Model
from keras.layers import ReLU, concatenate
import keras.backend as K

from keras import models
from scipy import ndimage

#------------------- data preprocessing --------------------------

def data_processing(flavor = 'N4cor_WarpedSegmentationPosteriors2', agumentation=False):
    '''
    a warpper function, to create an aggregated dataset to be used while training.

    flavor:         a segementation marker chosen after preprocessing the MRI scans. By default gray matter segementation are chosen.  
    agumentation:   a flag to decide the loaded dataset should contain agumented brain scans.    
    '''
    #loading datasets
    adni2 = available_datasets.load_ADNI2_data( x_range_from = 12, x_range_to = 181, 
                                                y_range_from = 13, y_range_to = 221,
                                                z_range_from = 0, z_range_to = 179, 
                                                flavor = flavor,
                                                aug=agumentation)

    aibl = available_datasets.load_AIBL_data(   x_range_from = 12, x_range_to = 181, 
                                                y_range_from = 13, y_range_to = 221,
                                                z_range_from = 0, z_range_to = 179, 
                                                flavor = flavor,
                                                aug=agumentation)

    edsd = available_datasets.load_EDSD_data(   x_range_from = 12, x_range_to = 181, 
                                                y_range_from = 13, y_range_to = 221,
                                                z_range_from = 0, z_range_to = 179, 
                                                flavor = flavor,
                                                aug=agumentation)

    delcode = available_datasets.load_DELCODE_data( x_range_from = 12, x_range_to = 181, 
                                                    y_range_from = 13, y_range_to = 221,
                                                    z_range_from = 0, z_range_to = 179, 
                                                    flavor = flavor,
                                                aug=agumentation)
                                                    
    adni3 = available_datasets.load_ADNI3_data( x_range_from = 12, x_range_to = 181, 
                                                y_range_from = 13, y_range_to = 221,
                                                z_range_from = 0, z_range_to = 179, 
                                                flavor = flavor,
                                                aug=False)                      #Always false, by default. For this is the test set.


    # combine datasets
    images = np.concatenate([adni2['images'], aibl['images'], edsd['images'], delcode['images']], axis=0)
    labels = np.concatenate([adni2['labels'], aibl['labels'], edsd['labels'], delcode['labels']], axis=0)

    labels = to_categorical(labels)
    groups = np.concatenate([ adni2['groups'], aibl['groups'], edsd['groups'], delcode['groups'] ], axis=0)
    covariates = np.concatenate([ adni2['covariates'], aibl['covariates'], edsd['covariates'], delcode['covariates'] ], axis=0)
    numfiles = labels.shape[0]

    #test set
    imagesT = adni3['images']
    groupsT = adni3['groups']
    labelsT = adni3['labels']
    labelsT = to_categorical(labelsT)
    covariatesT = adni3['covariates']

    del adni2; del aibl; del edsd; del delcode; del adni3  # free memory

    return [images, labels, groups, covariates, numfiles], [imagesT, groupsT, labelsT, covariatesT] 

def get_class_weights(labels, numfiles):
    '''
    define class weights to train a balanced model, 
    taken from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    Scaling by total/2 helps keep the loss to a similar magnitude. The sum of the weights of all examples stays the same.
    
    labels:     ground truth labels.
    numfiles:   total number of instances.
    '''
    neg, pos = np.bincount(labels[:, 1].astype(np.int_))
    weight_for_0 = (1 / neg)*(numfiles)/2.0 
    weight_for_1 = (1 / pos)*(numfiles)/2.0
    class_weights = {0: weight_for_0, 1: weight_for_1}
    print('Examples:    Total: {}    Positive: {} ({:.2f}% of total)'.format(
        numfiles, pos, 100 * pos / numfiles))
    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return class_weights


#-------------- model evaluation support functions --------------------------------

def save_train_plts(hist, path, validation_tracked=False):
    '''
    saves plots for traning and validation loss and accuracy, over the epochs. 
    the plots are saved in two formats - png and pdf

    hist:   a keras history object, obtained after model.fit() call. records evaluation metrics values at successive epochs. 
    path:   directory path to where the plots should be saved.
    validation_tracked:     a flag object. When turned on, enables tracking of validation metrics. 
    '''
    # get models statistics
    loss = hist.history['loss']
    acc = hist.history['acc']
    if validation_tracked:
        val_loss = hist.history['val_loss']
        val_acc = hist.history['val_acc']
    epochsr = range(len(loss))

    # plot the loss-vs-epoch graph
    plt.figure()
    plt.plot(epochsr, loss, 'bo', label='Training loss')
    if validation_tracked:
        plt.plot(epochsr, val_loss, 'b', label='Validation loss')
        plt.title('Training and Validation loss')
    else:
        plt.title('Training loss')
    plt.legend()
    plt.savefig(os.path.join(path, 'TrainLoss.png'))
    plt.savefig(os.path.join(path, 'TrainLoss.pdf'))

    # plot the accuracy-vs-epoch graph
    plt.figure()
    plt.plot(epochsr, acc, 'bo', label='Training acc')
    if validation_tracked:
        plt.plot(epochsr, val_acc, 'b', label='Validation acc')
    plt.title('Accuracy')
    plt.legend()
    #plt.show()
    plt.savefig(os.path.join(path, 'Accuracy.png'))
    plt.savefig(os.path.join(path, 'Accuracy.pdf'))

def ROC_AUC(labelsT,pred, path, dataset_marker='test'):
    '''
    Calculate area under the curve (AUC)
    Adapted from: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras

    labelsT:    the ground truth labels
    pred:       the predicted labels
    path:       directory path to where ROC curves have to be saved
    dataset_marker: a marker declaring portion of data set. ex Train, Validation or Test.
    '''
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    acc = dict()
    for i in range(2): # classes dummy vector: 0 - CN, 1 - MCI/AD
        fpr[i], tpr[i], _ = roc_curve(labelsT[:, i], pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr[1], tpr[1], color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic | {}-set'.format(dataset_marker))
    plt.legend(loc="lower right")
    #save
    plt.savefig(os.path.join(path, '{}_ROC_AUC.png'.format(dataset_marker)))    
    plt.savefig(os.path.join(path, '{}_ROC_AUC.pdf'.format(dataset_marker)))    


#------------- Model intialisation function calls -----------------------

def model_VGG(ip_shape, op_shape):
    '''
    declaring a VGG-Net model. 
    Paper: https://arxiv.org/pdf/1409.1556.pdf

    ip_shape: expected input shape
    op_shape: expected output shape
    '''
    # Setup 3D CNN model
    input_shape = ip_shape               #images.shape[1:]
    model = models.Sequential()

    # Convolution Layers
    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu',
                input_shape=input_shape, data_format='channels_last'))
    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.Conv3D(8, (3, 3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.SpatialDropout3D(rate = 0.5, data_format='channels_last'))

    # FC layers
    model.add(layers.Flatten())
    model.add(layers.Dropout(rate = 0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(rate = 0.1))
    model.add(layers.Dense(32, activation='relu', kernel_regularizer='l2'))
    model.add(layers.Dropout(rate = 0.1))
    model.add(layers.Dense(op_shape, kernel_regularizer='l2', activation='softmax'))    #labels.shape[1]

    return model

def model_Alex2(ip_shape, op_shape):
    '''
    declaring a AlexNet model. 
    Paper: https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    NOTE: adapted from Mahdi Osman Khan's prethesis work. 

    ip_shape: expected input shape
    op_shape: expected output shape
    '''
    # Setup 3D CNN model
    input_shape = ip_shape          #images.shape[1:]
    model = models.Sequential()

    # Convolution Layers
    model.add(layers.Conv3D(5, (3, 3, 3), padding='same', input_shape=input_shape, data_format='channels_last'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv3D(5, (3, 3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Activation('relu'))

    model.add(layers.Conv3D(5, (3, 3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate = 0.3))

    # FC layer
    model.add(layers.Flatten())
    model.add(layers.Dense(op_shape, activation='softmax'))
    
    return model

def model_Resnet(ip_shape, op_shape, activation='relu'):
    '''
    declaring a ResNet model. 
    Paper: https://arxiv.org/pdf/1512.03385.pdf
    NOTE: adapted from Mahdi Osman Khan's prethesis work. 

    ip_shape: expected input shape
    op_shape: expected output shape
    activation: activation (non-linearity) to be used in the model.
    '''
    # Setup 3D CNN model
    input_shape = ip_shape
    X=layers.Input(shape=input_shape)
    # Convolution Layers
    X1=layers.Conv3D(5, (3, 3, 3), padding='same', input_shape=input_shape, data_format='channels_last')(X)
    X1=layers.Conv3D(5, (3, 3, 3), padding='same')(X1)
    X1=layers.BatchNormalization()(X1)
    X1=layers.MaxPooling3D(pool_size=(2, 2, 2))(X1)
    X1=layers.Activation(activation)(X1)
    X1=layers.Conv3D(5, (3, 3, 3), padding='same')(X1)
    X1=layers.Conv3D(5, (3, 3, 3), padding='same')(X1)
    X1=layers.BatchNormalization()(X1)

    X2=layers.MaxPooling3D(pool_size=(2, 2, 2))(X)

    Y=layers.Add()([X1,X2])         #residual/skip connection


    Y=layers.MaxPooling3D(pool_size=(2, 2, 2))(Y)
    Y=layers.Activation(activation)(Y)


    Y1=layers.Conv3D(5, (3, 3, 3), padding='same')(Y)
    Y1=layers.Conv3D(5, (3, 3, 3), padding='same')(Y1)
    Y1=layers.Conv3D(5, (3, 3, 3), padding='same')(Y1)
    Y1=layers.BatchNormalization()(Y1)
    Y1=layers.MaxPooling3D(pool_size=(2, 2, 2))(Y1)
    Y1=layers.Activation(activation)(Y1)
    Y1=layers.Conv3D(5, (3, 3, 3), padding='same')(Y1)
    Y1=layers.Conv3D(5, (3, 3, 3), padding='same')(Y1)
    Y1=layers.Conv3D(5, (3, 3, 3), padding='same')(Y1)
    Y1=layers.BatchNormalization()(Y1)

    Y2=layers.MaxPooling3D(pool_size=(2, 2, 2))(Y)

    Z=layers.Add()([Y1,Y2])          #residual/skip connection

    Z=layers.MaxPooling3D(pool_size=(2, 2, 2))(Z)
    Z=layers.Activation(activation)(Z)
    Z=layers.Dropout(rate = 0.3)(Z)

    # FC layer
    Z=layers.Flatten()(Z)
    Z=layers.Dense(op_shape, activation='softmax')(Z)
    model = models.Model(inputs=X, outputs=Z)

    return model

def DenseNet(ip_shape, op_shape, filters = 3):
    '''
    declaring a DenseNet model. 
    Paper: https://arxiv.org/pdf/1608.06993.pdf
    Code adapted from https://towardsdatascience.com/creating-densenet-121-with-tensorflow-edbc08a956d8
    
    ip_shape: expected input shape
    op_shape: expected output shape
    filters: number of filters to be used
    '''

    #batch norm + relu + conv
    def bn_rl_conv(x,filters,kernel=1,strides=1):
        
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(filters, kernel, strides=strides,padding = 'same')(x)
        return x
    
    def dense_block(x, repetition=4):
        
        for _ in range(repetition):
            y = bn_rl_conv(x, filters=8)
            y = bn_rl_conv(y, filters=8, kernel=3)
            x = concatenate([y,x])
        return x
        
    def transition_layer(x):
        
        x = bn_rl_conv(x, K.int_shape(x)[-1] //2 )
        x = AveragePooling3D(2, strides = 2, padding = 'same')(x)
        return x
    


    input = Input(ip_shape)
    x = Conv3D(10, 7, strides = 2, padding = 'same')(input)
    x = MaxPooling3D(3, strides = 2, padding = 'same')(x)
    
    brc_in_blocks = [3,3]
    for repetition in brc_in_blocks:                      #[6,12,24,16]: 
        d = dense_block(x, repetition)
        x = transition_layer(d)
    
    #x = GlobalAveragePooling3D()(d)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(d)
    #Notice the input to pooling layer. d. this overwrites the last x variable,
    #and nullifies the transition layer computations done on last dense blocks output. 
    # i.e last transition layer is not connected to the graph 
    #TLDR: No transition layer after last dense block. 

    # FC layer
    x = layers.Flatten()(x)
    output = Dense(op_shape, activation = 'softmax')(x)
    
    model = Model(input, output)
    return model


#------------ data agumentation support functions --------------------------

'''
#A data agumentation procedure, zooms in a nd-array. Not used as was found to be too computationally expensive.    
def nd_zoom_in(ip, dim_code=[3,3,3]):
    #ip: inpt tensor - (Batch, Height, Width, Depth, Channels) (2001,179,169,208,1)
    #dim_code :  a cropping code for each dimention. [1: Crop from the end, 2: Crop from the begining, 3: Crop from both ends]  

    dim1_zoom = 1+ np.random.random()*0.025     #A random chosen zoom-in ratio. Is at most 2.5% in one dimention.
    dim2_zoom = 1+ np.random.random()*0.025     
    dim3_zoom = 1+ np.random.random()*0.025     

    #The zoom image volume, is at 'most' zommed 7%. Possibly lesser in practice.
    #zoomed = ndimage.zoom(ip, (1, dim1_zoom, dim2_zoom, dim3_zoom, 1))    
    zoomed = ndimage.zoom(ip, (dim1_zoom, dim2_zoom, dim3_zoom, 1))
    #print(zoomed.shape)

    #Difference between 3d vol shape, before and after zooming 
    diff = np.subtract(zoomed.shape , ip.shape)

    crop_type = {'batch':':', 'channel':':'}
    dimentions= ['height','width','depth']   
    for i,(code,dim) in enumerate(zip(dim_code,dimentions)):
        if code==1:
            #Crop from the end
            crop_type[dim] =  ':ip.shape['+str(i+1)+']' 
        elif code==2:
            #Crop from the begining
            crop_type[dim] =  'diff['+str(i+1)+']:' 
        elif code==3:
            #Crop from both end
            if (diff[i+1]%2 == 0) and (diff[i+1]>0):
                cr = int(diff[i+1]/2)
                crop_type[dim] =   str(cr)+':'+'zoomed.shape['+str(i+1)+'] -'+ str(cr) 
            else:
                #print('Unable to crop from both end. Default to cropping from the end.')
                crop_type[dim] =  ':ip.shape['+str(i+1)+']' 
        else:
            #print('Invalid Cropping Code. Default to cropping from the end.')
            #print('Input image returned as is')
            return ip

    #op = eval('zoomed[' + crop_type['batch'] +','+ crop_type['height'] +','+ crop_type['width'] +','+ crop_type['depth'] +','+ crop_type['channel'] + ']') 
    op = eval('zoomed[' +  crop_type['height'] +','+ crop_type['width'] +','+ crop_type['depth'] +','+ crop_type['channel'] + ']') 


    return op
'''

def shift(arr, num, axis, fill_value=0):
  ''' defines function for simple data augmentation (translation of 2 vx in each x/y/z direction)  
      adapted from https://stackoverflow.com/a/42642326
    
      arr:  a ndarray to be shifted along axes
      num:  in interger number, quatifying the number of voxels a volume has to shifted. 
            the sign signifies the direction of shift along the axis. 
      axis: the length, breadth, depth (or coronal, axial, sagittal) axis along which shift takes place.
      fill_value:   a replacement value, for the voxles where shift took place.  
  '''

  result = np.empty_like(arr)
  if (axis==1):
    if num > 0:
        result[:num, :, :] = fill_value
        result[num:, :, :] = arr[ :-num, :, :]
    elif num < 0:
        result[num:, :, :] = fill_value
        result[:num, :, :] = arr[ -num:, :, :]
    else:
        result[:] = arr
  elif (axis==2):
    if num > 0:
        result[ :, :num, :] = fill_value
        result[ :, num:, :] = arr[ :, :-num, :]
    elif num < 0:
        result[ :, num:, :] = fill_value
        result[ :, :num, :] = arr[ :, -num:, :]
    else:
        result[:] = arr
  elif (axis==3):
    if num > 0:
        result[ :, :, :num] = fill_value
        result[ :, :, num:] = arr[ :, :, :-num]
    elif num < 0:
        result[ :, :, num:] = fill_value
        result[ :, :, :num] = arr[ :, :, -num:]
    else:
        result[:] = arr
  else:
    print('Invalid axis chosen for shift. Can only shift around Height, Widht, Depth axis. Format BHWDC')
    return None

  return result  

def data_aug_inplace(trainigdata, flip=True):
    '''
    a data-augmentation wrapper function around the shift procedure.

    trainigdata:    the nd.array of dataset to be augmented.
    flip:           a flag variable, when turned on, enables flipping the order of elements along an axis. 
    '''
    #axis1
    if np.random.choice([0, 1]):       #50% chance of shift along an axis
        shift_side = np.random.choice([-1, 1])     #50% chance, in deciding the direction of shift 
        # 7 voxels, is an heurisitic choice. With an MRI volume size of 179x169x208 voxels, 
        # 7/169 => (atmost) approx 5% volume shift along one direction. 
        trainigdata =  shift(trainigdata, axis=1, num=7*shift_side)   
    #axis2
    if np.random.choice([0, 1]):
        shift_side = np.random.choice([-1, 1])
        trainigdata =  shift(trainigdata, axis=2, num=7*shift_side)
    #axis3
    if np.random.choice([0, 1]):
        shift_side = np.random.choice([-1, 1])
        trainigdata =  shift(trainigdata, axis=3, num=7*shift_side)
    
    if flip:     #L/R flipping
        if np.random.choice([0, 1]):     #if flip turned on, 50% chance of flippping
            trainigdata = np.fliplr(trainigdata)
    

    return trainigdata

