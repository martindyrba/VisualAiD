'''
A script for creating the relevance map slices, for a chosen set of representative samples.
'''

import numpy as np
import rmap_util
import available_datasets
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import warnings
warnings.filterwarnings('ignore')

#hyperparameters used for visualizing relevance maps
q = 0.9999
overlay_alpha = 0.5
overlay_alpha_threshold= 0.2

#path initialisation and saving config
data_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/data/'
all_experiments_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments'
save_path = '/data_dzne_archiv2/Studien/Deep_Learning_Visualization/git-code/demenzerkennung/Devesh/experiments/relevance3'
rmap_util.dir_check(save_path)
rmap_util.save_config(save_path, q, overlay_alpha, overlay_alpha_threshold)

#A list of possible relevance propagation methods
# initialised as a tuple with (method, params, label)
methods = [ 
                #("deconvnet",            {},                      "Deconvnet"),
                #("guided_backprop",      {},                      "Guided Backprop"),
                #("deep_taylor.bounded",  {"low": -1, "high": 1},  "DeepTaylor"),
                #("input_t_gradient",     {},                      "Input * Gradient"),
                #("lrp.z",                {},                      "LRP-Z"),
                #("lrp.epsilon",          {"epsilon": 1},          "LRP-epsilon"),
                #("lrp.alpha_1_beta_0",   {},                      "LRP-alpha1beta0"),
                ("lrp.sequential_preset_a", {"neuron_selection_mode": "index", "epsilon": 1e-10}, "LRP-CMPalpha1beta0"), #"neuron_selection_mode": "index" # LRP CMP rule taken from https://github.com/berleon/when-explanations-lie/blob/master/when_explanations_lie.py
                #("lrp.sequential_preset_b", {"neuron_selection_mode": "index", "epsilon": 1e-10}, "LRP-CMPalpha2beta1"), 
    ]

#Chosen set of representative samples, from different cohorts of datasets. 
AD_samples =  [ 'ADNI_t1linear/AD/AD_4910_N4cor_WarpedSegmentationPosteriors2.nii.gz',
                'DELCODE_t1linear/abea7cf25N4cor_WarpedSegmentationPosteriors2.nii.gz',
                'EDSD_t1linear/ROS3T_AD008_N4cor_WarpedSegmentationPosteriors2.nii.gz'   
             ]
CN_samples =  ['DELCODE_t1linear/953d43cefN4cor_WarpedSegmentationPosteriors2.nii.gz',
               'EDSD_t1linear/ROS3T_HC016_N4cor_WarpedSegmentationPosteriors2.nii.gz'  
                ]

#List of experiment timestamp-ids and their names.
experiments_list =  ['261022_164054',   '271022_164133',    '281122_144817',    '251122_180422',
                    '021122_104936',    '031122_101114',    '071122_094130',    '101122_150350' ]   
model_type =        ['ResNet-ua',       'ResNet-a',         'DenseNet-ua',      'DenseNet-a',
                    'AlexNet-a',        'Alexnet-ua',       'VGG-ua',           'VGG-a']            

folds = ['cv2']               #Chosen because - highest average binary(cn vs ad/mci) test accuracy on this fold.

#creating a nested dict of innvestigate anaylzers
exp_dict = rmap_util.create_anaylzer_dict(experiments_list, model_type, methods, folds, all_experiments_path)


selected_neuron = 1
flavour = 'N4cor_WarpedSegmentationPosteriors2.nii.gz'
for sample in AD_samples+CN_samples:   #AD_samples+CN_samples:
    id = rmap_util.remove_last_occurrence(sample.split('/')[-1].replace(flavour,'') , '_') 
    dataset = sample.split('/')[0].replace('_t1linear','')
    sample_save_path = os.path.join(save_path, id)
    os.makedirs(sample_save_path, exist_ok=True)

    #path were a processed GM file for a sample is saved.
    im_path = data_path+sample
    #reading the file
    img = available_datasets.read_nifti_data(im_path,
                                             x_range_from = 12, x_range_to = 181, 
                                             y_range_from = 13, y_range_to = 221, 
                                             z_range_from = 0, z_range_to = 179, 
                                             minmaxscaling = True, aug=False)
    
    #tf_im = (tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32),-1),0))
    im = np.expand_dims(np.expand_dims(img,-1),0)


    for exp_name in exp_dict.keys():
        for method in exp_dict[list(exp_dict.keys())[0]].keys():
            try:  #getting the relevance values, using an analyzer object.
                if 'CMP' in method:
                    a = exp_dict[exp_name][method].analyze(im,neuron_selection=selected_neuron)
                else:
                    a = exp_dict[exp_name][method].analyze(im)
            except:
                print(exp_name,'\t', method)
                continue
            
            a = rmap_util.scale_relevance_map(r_map=a, clipping_threshold=1, quantile=q)
            #Save activation to nii, where a in [-1,1] 
            rmap_util.activation2nifti(a ,im_path, id, dataset, exp_name, method, sample_save_path)   

            #printing and saving some slices of relevances values (plotted of underlying brain scan),
            a = np.squeeze(a)
            rmap_util.print_image_slices(a, im, exp_name, method, overlay_alpha, overlay_alpha_threshold, sample_save_path)