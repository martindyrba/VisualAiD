'''
A script for utility functionalities created for relevance/activations management.     
'''

import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import util
from keras import models
import innvestigate.utils as iutils
import innvestigate
import nibabel as nib

import warnings
warnings.filterwarnings('ignore')

def dir_check(save_path):
    '''
    Create a director. And if it already exists, check if it is empty or not.

    save_path: a directory path
    '''

    os.makedirs(save_path, exist_ok=True)
    dir = os.listdir(save_path)
    if len(dir) == 0:
        #Empty directory
        pass
    else:
        #Not empty directory
        raise SystemExit('Not empty relevance directory')

def save_config(save_path, q , overlay_alpha , overlay_alpha_threshold):
    '''
    Saves a config(.txt) file, containing hyperparameter setings that were used for creating relevance maps.

    save_path: a directory path
    q: Scaling Quantile
    overlay_alpha: Overlay Alpha shown
    overlay_alpha_threshold: Overlay Alpha thresholded
    ''' 
    config_dict = {'Scaling Quantile': q, 'Overlay Alpha shown': overlay_alpha, 'Overlay Alpha thresholded': overlay_alpha_threshold}

    with open( os.path.join(save_path,'config.txt'), 'w') as f:
        f.write(str(config_dict))


def remove_last_occurrence(string, substring):
    return ''.join(string.rsplit(substring, 1))


# Intialising the colormap to be used.
overlay_colormap = cm.get_cmap('RdYlGn_r')



def activation2nifti( a, im_path, sample_id, sample_dataset, cnn_experiment_name, relevance_method_name, save_path,
                    x_range_from=12, x_range_to=181, y_range_from=13, y_range_to=221, z_range_from=0, z_range_to=179):
    '''
    Converts and saves, an activation volume to a nifti file. 

    a: activation volume to be saved. Assumes in range [-1, 1]
    im_path:  path to any sample MRI scan with nifti extention. Used for header.
    sample_id: str rid. a unique marker for the sample MRI scan.
    sample_dataset: str marker of the dataset from which MRI sample was taken. ex ADNI or delcode.
    cnn_experiment_name: str marker of the 3d cnn model used for creating this activation volume
    relevance_method_name: name of the relevance propogation method used for creating this activation volume
    save_path: directory path
    '''
    
    #load a sample MRI scan to be used for header. 
    hipp_nifti = nib.load(im_path)                                      # assume it is already 32bit float format
    new_data = np.zeros(hipp_nifti.shape, dtype=np.float32) 

    a = np.squeeze(a)
    a = np.flip(a)                                                      # flip all positions
    a = np.transpose( a , (1, 2, 0))                        # reorder dimensions from coronal view z*x*y back to x*y*z
    
    
    #print('saving relevance maps:\n  Sample/Dataset: {}/{}, \n  Model:{} \t Relevance method:{}'.format(sample_id,sample_dataset,cnn_experiment_name, relevance_method_name)) 
    
    #assign                
    new_data[x_range_from:x_range_to, y_range_from:y_range_to, z_range_from:z_range_to]   = a
    nifti = nib.Nifti1Image(new_data, hipp_nifti.affine, hipp_nifti.header)
    
    #saving
    nifti.to_filename(os.path.join(save_path,'{}_{}_ActVol.nii'.format(cnn_experiment_name, relevance_method_name))) 
    

def create_anaylzer_dict(experiments_list, model_type, methods, folds, all_experiments_path):
    '''
    Creates a nested dictionary of innvestigate analyzers, for each model architecture and relevance propogation method.

    experiments_list:   list of relevant experiments where 3d-cnn models were trained. Each experiment in str(datetime) format.    
    model_type:         list of unique markers for each relevant experiments, to identify the experiments in human readable manner.
    methods:            list of relevance propogation methods. Each element is a tuple like (innvestigate.RelevanceMethod, required arguments, user-defined method identifiers)
    folds:              list of cross-validation folds chosen for creating relevance activations on.
    '''

    exp_dict ={} 

    #loop over all 3d-cnn model experiments.
    for exp,m_type in zip(experiments_list,model_type):       

        #if '-a' in m_type:    #Take this condition out when want to include agumented data models in the relevance testing.
        #    continue

        #To avoid empty folders    
        experiment_path = os.path.join(all_experiments_path, exp)
        fold_path = os.path.join(experiment_path, folds[0] )
        if not os.path.exists(fold_path):
            continue

        #initialisations
        model_code = 'model_{}-vgg.best.hdf5'.format(folds[0])
        model_path = os.path.join(fold_path, model_code)

        ip_shape = (179,169,208,1)
        op_shape = 2
        
        model = None

        #loading models with random weights
        if 'ResNet' in m_type:
            model =  util.model_Resnet(ip_shape,op_shape)
        elif 'DenseNet' in m_type:
            model =  util.DenseNet(ip_shape,op_shape)
        elif 'AlexNet' in m_type:
            model =  util.model_Alex2(ip_shape,op_shape)
        else:
            model = util.model_VGG(ip_shape, op_shape)


        try:
            model = models.load_model(model_path)                               # loading models with trained weights
            model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)       # removing softmax layer from the end, needed for relevance maps via innvestigate

            # In case model_wo_softmax crashes
            #From: https://github.com/martindyrba/DeepLearningInteractiveVis/blob/demo/x_extract_hippocampus_relevance_lrpCMP.ipynb
            #model.layers[-1].activation = tf.keras.activations.linear
            #model.save('tmp_wo_softmax.hdf5')
            #model_wo_softmax = models.load_model('tmp_wo_softmax.hdf5')
            #os.remove('tmp_wo_softmax.hdf5')
            #model_wo_softmax.summary()
        except:
            print(m_type)                                                       # prints name of model for which removing softmax layer was not possible 
            continue
        
        #creating a dict of all the relevance analyzers (methods)    
        methods_dict = {} 
        for method in methods:
            analyzer = innvestigate.create_analyzer(method[0], model_wo_softmax, **method[1])
            # Some analyzers require training.
            #analyzer.fit(tf_im, batch_size=30, verbose=1)      
            methods_dict[method[2]] = analyzer

        #creates nested dict of all relenave methods, inside each experiment
        exp_dict[m_type] = methods_dict  

    return exp_dict



# Adjusted global color palette for ColorBar annotation, because bokeh does not support some palettes by default:
def overlay2rgba(relevance_map, alpha=0.5, alpha_threshold=0.2):
    """
    Converts the 3D relevance map to RGBA.

    relevance_map: numpy.ndarray relevance_map: The 3D relevance map.
    alpha: the transparency/the value for the alpha channel.
    alpha_threshold: min threshold for showing relevance values.
    """
    # assume map to be in range of -1..1 with 0 for hidden content
    alpha_mask = np.zeros_like(relevance_map)
    alpha_mask[np.abs(relevance_map) > alpha_threshold] = alpha          # final transparency of visible content
    relevance_map = relevance_map / 2 + 0.5  # range 0-1 float
    ovl = np.uint8(overlay_colormap(relevance_map) * 255)  # cm translates range 0 - 255 uint to rgba array
    ovl[:, :, 3] = np.uint8(alpha_mask * 255)          #ovl[:, :, 3]              # replace alpha channel (fourth dim) with calculated values
    #ret = ovl.view("uint32").reshape(ovl.shape[:2])    #ovl.shape[:2]             # convert to 3D array of uint32
    ret = Image.fromarray(ovl)
   
    return ret


def scale_relevance_map(r_map, clipping_threshold=1, quantile=0.9999):
    """
    Clips the relevance map to given threshold and adjusts it to range -1...1 float.

    relevance_map: relevance activation volume
    clipping_threshold: max value to be plotted, larger values will be set to this value
    quantile: quantile chosen to scale the data. 
    """
   #if debug: print("Called scale_relevance_map()")
    #    r_map = np.copy(relevance_map)  # leave original object unmodified.
    
    # perform intensity normalization
    adaptive_relevance_scaling=True

    if adaptive_relevance_scaling:
        scale = np.quantile(np.absolute(r_map), quantile)  #.9999
    else:
        scale = 1/500     # multiply by 500
    if scale != 0:  # fallback if quantile returns zero: directly use abs max instead
        r_map = (r_map / scale)  # rescale range
    # corresponding to vmax in plt.imshow; vmin=-vmax used here
    # value derived empirically here from the histogram of relevance maps
    r_map[r_map > clipping_threshold] = clipping_threshold  # clipping of positive values
    r_map[r_map < -clipping_threshold] = -clipping_threshold  # clipping of negative values
    r_map = r_map / clipping_threshold  # final range: -1 to 1 float
    return r_map


def save_frequency_plots(a, exp_name,method, save_path):
    '''
    creates and saves, a frequency histogram for a given activation volume.

    a:          relevance maps (3D volume). Assumed to be clippped between [-1,1].
    exp_name:   experiment name. An identifier for the type of 3D-CNN model architecture is being used.
    method:     marker for relevance propogation algorithm, ex: LRP-epsilon  
    save_path:  a directory path.  
    '''
    
    plt.figure()
    fig, ax = plt.subplots()
    ax.hist(np.reshape(a, -1), bins=50, range=(-2.5,2.5))
    fig.gca().set_yscale('log')
    fig.gca().set(title='Frequency Histogram', ylabel='Frequency')
    ax.text(0.9,0.9,        
            'Max: {}'.format(np.max(a)),                        #Prints the maximum activation value found in the volume.
             ha='right', va='top',
            transform=ax.transAxes 
            )
    plt.savefig(os.path.join(save_path,'{}_{}_FreqHist.png'.format(exp_name,method)))
    plt.close()


def print_image_slices(a,img,exp_name,method, overlay_alpha=0.5, overlay_alpha_threshold=0.2, save_path=''):
    '''
    saves activation/relevance values over slices of brain scans as an image.

    a:          relevance maps (3D volume). Assumed to be clippped between [-1,1].
    img:        underlying input image for which activation maps a, are created.
    exp_name:   experiment name. An identifier for the type of 3D-CNN model architecture is being used.
    method:     marker for relevance propogation algorithm, ex: LRP-epsilon    
    overlay_alpha: the transparency/the value for the alpha channel.
    overlay_alpha_threshold: min threshold for showing relevance values.
    save_path:  a directory path
    '''

    #1 - create and save frequeny histogram (to find the distribution of) the avtivation volume.
    save_frequency_plots(a, exp_name,method, save_path)

    #2 - Create 2D relvance visualisations. Avtivations ploted over coronal slices.  
    # RGBA o/p (4 channel) - RGB brain image, activations placed in the alpha channel
    images_per_row = 10                                             # only ten evenly spaced sclices out of volume depth of 111 units.
    jump = (a.shape[2] // images_per_row)
    for row in range(images_per_row):
        plt.figure()
        plt.imshow((img[0, :,:, row*jump, 0]), cmap='gray')
        #plt.imshow(prepare_overlay((a[:,:, row*jump]), rescale_threshold=1, alpha_treshold = 0.4, overlay_alpha = 0.5))   #Older method of creating overlays 
        plt.imshow(overlay2rgba(a[:,:, row*jump], alpha=overlay_alpha, alpha_threshold=overlay_alpha_threshold))  
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
        plt.gca().set(title=method, ylabel='Smoothed & thresholded', xlabel=('cor idx %d' % (row*jump)));
        plt.savefig(os.path.join(save_path,'{}__{}__{}.png'.format(exp_name,method,row*jump)))
        plt.close()