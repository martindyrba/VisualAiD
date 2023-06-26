import pandas as pd
import glob
import re
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import util 

#%%
def load_ADNI2_data(x_range_from = 15, x_range_to = 175, y_range_from = 90, y_range_to = 130, z_range_from = 15, z_range_to = 175, flavor = 'N4cor_WarpedSegmentationPosteriors2', minmaxscaling = True, debug = False, aug=False):
    """
    Load the ADNI-GO/2 dataset and return it as numpy array.
    
    Typically, we will use a reduced FOV to save required memory size.
    
    # Clinica range for cropped image:
    x_range_from = 12; x_range_to = 181  # R - L
    y_range_from = 13; y_range_to = 221  # P - A
    z_range_from = 0; z_range_to = 179   # I - S
    
    :return: 

    Parameters
    ----------
    x_range_from : int >= 0, optional
        Input file saggital slice start. The default is 15.
    x_range_to : int >= 0, optional
        Input file saggital slice end. The default is 175.
    y_range_from : int >= 0, optional
        Input file coronal slice start. The default is 90.
    y_range_to : int >= 0, optional
        Input file coronal slice end. The default is 130.
    z_range_from : int >= 0, optional
        Input file axial slice start. The default is 15.
    z_range_to : int >= 0, optional
        Input file axial slice end. The default is 175.
    flavor : str, optional
        The input file name suffix. Defines which data to load, e.g. 'N4cor_WarpedSegmentationPosteriors2'.
    minmaxscaling : bool, optional
        Apply scaling to adjust minimum and maximum intensity to the range 0 and 1. The default is True.
    debug : bool, optional
        Display additional information for participant's file and spreadsheet matching. The default is False.

    Returns
    -------
    dict
        A dictionary containing the images as numpy array, the binary labels, the original labels, the covariates.

    """

    # Import data from Excel sheet
    df = pd.read_excel('/mnt/ssd1/dyrba/ADNI combined.xlsx', sheet_name='aal_volumes')
    
    sid = df['RID'] #'subject_ID'
    grp = df['Group at scan date (1=CN, 2=EMCI, 3=LMCI, 4=AD, 5=SMC)'].replace(3, 2).replace(4, 3)
    age = df['Age at scan']
    sex = df['Sex (1=female)']
    #tiv = df['TIV_CAT12']
    field = df['MRI_Field_Strength']
    #amybin = df['Amy SUVR Pos']
    grpbin = (grp > 1) # 1=CN, ...
    
    
    # Scan for nifti file names
    dataAD = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/AD/*_'+flavor+'.nii.gz'))
    dataLMCI = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/LMCI/*_'+flavor+'.nii.gz'))
    #dataEMCI = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/EMCI/*_'+flavor+'.nii.gz'))
    dataCN = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/CN/*_'+flavor+'.nii.gz'))
    #dataADNI3 = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/ADNI3/*_'+flavor+'.nii.gz'))
    dataFiles = dataAD + dataLMCI + dataCN #+ dataEMCI + dataADNI3
    numfiles = len(dataFiles)
    print('Found ', str(numfiles), ' nifti files')
    
    
    # Match covariate information
    cov_idx = [-1] * numfiles
    print('Matching covariates for loaded files ...')
    for i,id in enumerate(sid):
      p = [j for j,x in enumerate(dataFiles) if re.search('_%04d_' % id, x)] # translate ID numbers to four-digit numbers, get both index and filename
      if len(p)==0:
        if debug: print('Did not find %04d' % id) # did not find Excel sheet subject ID in loaded file selection
      else:
        if debug: print('Found %04d in %s: %s' % (id, p[0], dataFiles[p[0]]))
        cov_idx[p[0]] = i # store Excel index i for data file index p[0]
    print('Checking for scans not found in Excel sheet: ', sum(x<0 for x in cov_idx))
    
    labels = pd.DataFrame({'Group':grpbin}).iloc[cov_idx, :]
    grps = pd.DataFrame({'Group':grp, 'RID': util.add_marker('ADNI2_', sid) }).iloc[cov_idx, :]
    covariates = pd.DataFrame({'Age':age/100, 'Sex':sex, 'FS':(field-1.5)/1.5}).iloc[cov_idx, :]
    print("Covariates data frame size : ", covariates.shape)
    print(covariates.head())
    covariates = covariates.to_numpy(dtype=np.float32) # convert dataframe to nparray with 32bit types
    
    
    # Actually load nifti files into array
    print('Loading files...')
    # 1. dimension: subject
    # 2. dimension: img row
    # 3. dimension: img col
    # 4. dimension: img depth
    # 5. dimension: img channel
    images = np.zeros((numfiles, z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from, 1),
                      dtype=np.float32) # numfiles× z × x × y ×1; avoid 64bit types
    for i in range(numfiles):   # for loop over files and load
        if (i % 50 == 0):
            print('Loading file %d of %d' % (i+1, numfiles))
        images[i, :,:,:, 0] = read_nifti_data(dataFiles[i], x_range_from, x_range_to, y_range_from, y_range_to, z_range_from, z_range_to, minmaxscaling, aug)
    print('Successfully loaded files')
    print('Image array size: ', images.shape)
    
    return {'images': images, 'labels': labels, 'groups': grps, 'covariates': covariates}


#%%
def load_ADNI3_data(x_range_from = 15, x_range_to = 175, y_range_from = 90, y_range_to = 130, z_range_from = 15, z_range_to = 175, flavor = 'N4cor_WarpedSegmentationPosteriors2', minmaxscaling = True, debug = False, aug=False):
    """
    Load the ADNI-3 dataset and return it as numpy array.
    
    Typically, we will use a reduced FOV to save required memory size.
    
    # Clinica range for cropped image:
    x_range_from = 12; x_range_to = 181  # R - L
    y_range_from = 13; y_range_to = 221  # P - A
    z_range_from = 0; z_range_to = 179   # I - S
    
    :return: 

    Parameters
    ----------
    x_range_from : int >= 0, optional
        Input file saggital slice start. The default is 15.
    x_range_to : int >= 0, optional
        Input file saggital slice end. The default is 175.
    y_range_from : int >= 0, optional
        Input file coronal slice start. The default is 90.
    y_range_to : int >= 0, optional
        Input file coronal slice end. The default is 130.
    z_range_from : int >= 0, optional
        Input file axial slice start. The default is 15.
    z_range_to : int >= 0, optional
        Input file axial slice end. The default is 175.
    flavor : str, optional
        The input file name suffix. Defines which data to load, e.g. 'N4cor_WarpedSegmentationPosteriors2'.
    minmaxscaling : bool, optional
        Apply scaling to adjust minimum and maximum intensity to the range 0 and 1. The default is True.
    debug : bool, optional
        Display additional information for participant's file and spreadsheet matching. The default is False.

    Returns
    -------
    dict
        A dictionary containing the images as numpy array, the binary labels, the original labels, the covariates.

    """

    # Import data from Excel sheet
    df = pd.read_excel('/mnt/ssd1/dyrba/ADNI combined.xlsx', sheet_name='aal_volumes')
    
    sid = df['RID'] #'subject_ID'
    grp = df['Group at scan date (1=CN, 2=EMCI, 3=LMCI, 4=AD, 5=SMC)'].replace(3, 2).replace(4, 3)
    age = df['Age at scan']
    sex = df['Sex (1=female)']
    #tiv = df['TIV_CAT12']
    field = df['MRI_Field_Strength']
    #amybin = df['Amy SUVR Pos']
    grpbin = (grp > 1) # 1=CN, ...
    
    
    # Scan for nifti file names
    #dataAD = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/AD/*_'+flavor+'.nii.gz'))
    #dataLMCI = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/LMCI/*_'+flavor+'.nii.gz'))
    #dataEMCI = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/EMCI/*_'+flavor+'.nii.gz'))
    #dataCN = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/CN/*_'+flavor+'.nii.gz'))
    dataADNI3 = sorted(glob.glob('/mnt/ssd1/dyrba/ADNI_t1linear/ADNI3/*_'+flavor+'.nii.gz'))
    dataFiles = dataADNI3 # dataAD + dataLMCI + dataCN + dataEMCI + dataADNI3
    numfiles = len(dataFiles)
    print('Found ', str(numfiles), ' nifti files')
    
    
    # Match covariate information
    cov_idx = [-1] * numfiles
    print('Matching covariates for loaded files ...')
    for i,id in enumerate(sid):
      p = [j for j,x in enumerate(dataFiles) if re.search('_%04d_' % id, x)] # translate ID numbers to four-digit numbers, get both index and filename
      if len(p)==0:
        if debug: print('Did not find %04d' % id) # did not find Excel sheet subject ID in loaded file selection
      else:
        if debug: print('Found %04d in %s: %s' % (id, p[0], dataFiles[p[0]]))
        cov_idx[p[0]] = i # store Excel index i for data file index p[0]
    print('Checking for scans not found in Excel sheet: ', sum(x<0 for x in cov_idx))
    
    labels = pd.DataFrame({'Group':grpbin}).iloc[cov_idx, :]
    grps = pd.DataFrame({'Group':grp, 'RID':util.add_marker('ADNI3_', sid)}).iloc[cov_idx, :]
    covariates = pd.DataFrame({'Age':age/100, 'Sex':sex, 'FS':(field-1.5)/1.5}).iloc[cov_idx, :]
    print("Covariates data frame size : ", covariates.shape)
    print(covariates.head())
    covariates = covariates.to_numpy(dtype=np.float32) # convert dataframe to nparray with 32bit types
    
    
    # Actually load nifti files into array
    print('Loading files...')
    # 1. dimension: subject
    # 2. dimension: img row
    # 3. dimension: img col
    # 4. dimension: img depth
    # 5. dimension: img channel
    images = np.zeros((numfiles, z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from, 1),
                      dtype=np.float32) # numfiles× z × x × y ×1; avoid 64bit types
    for i in range(numfiles):   # for loop over files and load
        if (i % 50 == 0):
            print('Loading file %d of %d' % (i+1, numfiles))
        images[i, :,:,:, 0] = read_nifti_data(dataFiles[i], x_range_from, x_range_to, y_range_from, y_range_to, z_range_from, z_range_to, minmaxscaling, aug)
    print('Successfully loaded files')
    print('Image array size: ', images.shape)
    
    return {'images': images, 'labels': labels, 'groups': grps, 'covariates': covariates}


#%%
def load_AIBL_data(x_range_from = 15, x_range_to = 175, y_range_from = 90, y_range_to = 130, z_range_from = 15, z_range_to = 175, flavor = 'N4cor_WarpedSegmentationPosteriors2', minmaxscaling = True, debug = False, aug=False):
    """
    Load the AIBL dataset and return it as numpy array.
    
    Typically, we will use a reduced FOV to save required memory size.
    
    # Clinica range for cropped image:
    x_range_from = 12; x_range_to = 181  # R - L
    y_range_from = 13; y_range_to = 221  # P - A
    z_range_from = 0; z_range_to = 179   # I - S
    
    :return: 

    Parameters
    ----------
    x_range_from : int >= 0, optional
        Input file saggital slice start. The default is 15.
    x_range_to : int >= 0, optional
        Input file saggital slice end. The default is 175.
    y_range_from : int >= 0, optional
        Input file coronal slice start. The default is 90.
    y_range_to : int >= 0, optional
        Input file coronal slice end. The default is 130.
    z_range_from : int >= 0, optional
        Input file axial slice start. The default is 15.
    z_range_to : int >= 0, optional
        Input file axial slice end. The default is 175.
    flavor : str, optional
        The input file name suffix. Defines which data to load, e.g. 'N4cor_WarpedSegmentationPosteriors2'.
    minmaxscaling : bool, optional
        Apply scaling to adjust minimum and maximum intensity to the range 0 and 1. The default is True.
    debug : bool, optional
        Display additional information for participant's file and spreadsheet matching. The default is False.

    Returns
    -------
    dict
        A dictionary containing the images as numpy array, the binary labels, the original labels, the covariates.

    """

    # Import data from Excel sheet
    df = pd.read_excel('/mnt/ssd1/dyrba/aibl_ptdemog_final.xlsx', sheet_name='aibl_ptdemog_final')
    sid = df['RID']
    grp = df['DXCURREN']
    age = df['age']
    sex = df['PTGENDER(1=Female)']
    #tiv = df['Total'] # TIV
    field = df['field_strength']
    grpbin = (grp > 1) # 1=CN, ...
        
    
    # Scan for nifti file names
    dataAIBL = sorted(glob.glob('/mnt/ssd1/dyrba/AIBL_t1linear/*'+flavor+'.nii.gz'))
    dataFiles = dataAIBL
    numfiles = len(dataFiles)
    print('Found ', str(numfiles), ' nifti files')
    
    
    # Match covariate information
    cov_idx = [-1] * numfiles # list; array: np.full((numfiles, 1), -1, dtype=int)
    print('Matching covariates for loaded files ...')
    for i,id in enumerate(sid):
      p = [j for j,x in enumerate(dataFiles) if re.search('%d_' % id, x)] # extract ID numbers from filename, translate to Excel row index
      if len(p)==0:
        if debug: print('Did not find %04d' % id) # did not find Excel sheet subject ID in loaded file selection
      else:
        if debug: print('Found %04d in %s: %s' % (id, p[0], dataFiles[p[0]]))
        cov_idx[p[0]] = i # store Excel index i for data file index p[0]
    print('Checking for scans not found in Excel sheet: ', sum(x<0 for x in cov_idx))
    
    labels = pd.DataFrame({'Group':grpbin}).iloc[cov_idx, :]
    grps = pd.DataFrame({'Group':grp, 'RID':util.add_marker('AIBL_', sid)}).iloc[cov_idx, :]
    covariates = pd.DataFrame({'Age':age/100, 'Sex':sex, 'FS':(field-1.5)/1.5}).iloc[cov_idx, :]
    print("Covariates data frame size : ", covariates.shape)
    print(covariates.head())
    covariates = covariates.to_numpy(dtype=np.float32) # convert dataframe to nparray with 32bit types
    
    
    # Actually load nifti files into array
    print('Loading files...')
    # 1. dimension: subject
    # 2. dimension: img row
    # 3. dimension: img col
    # 4. dimension: img depth
    # 5. dimension: img channel
    images = np.zeros((numfiles, z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from, 1),
                      dtype=np.float32) # numfiles× z × x × y ×1; avoid 64bit types
    for i in range(numfiles):   # for loop over files and load
        if (i % 50 == 0):
            print('Loading file %d of %d' % (i+1, numfiles))
        images[i, :,:,:, 0] = read_nifti_data(dataFiles[i], x_range_from, x_range_to, y_range_from, y_range_to, z_range_from, z_range_to, minmaxscaling, aug)
    print('Successfully loaded files')
    print('Image array size: ', images.shape)
    
    return {'images': images, 'labels': labels, 'groups': grps, 'covariates': covariates}


#%%
def load_DELCODE_data(x_range_from = 15, x_range_to = 175, y_range_from = 90, y_range_to = 130, z_range_from = 15, z_range_to = 175, flavor = 'N4cor_WarpedSegmentationPosteriors2', minmaxscaling = True, debug = False, aug=False):
    """
    Load the DELCODE dataset and return it as numpy array.
    
    Typically, we will use a reduced FOV to save required memory size.
    
    # Clinica range for cropped image:
    x_range_from = 12; x_range_to = 181  # R - L
    y_range_from = 13; y_range_to = 221  # P - A
    z_range_from = 0; z_range_to = 179   # I - S
    
    :return: 

    Parameters
    ----------
    x_range_from : int >= 0, optional
        Input file saggital slice start. The default is 15.
    x_range_to : int >= 0, optional
        Input file saggital slice end. The default is 175.
    y_range_from : int >= 0, optional
        Input file coronal slice start. The default is 90.
    y_range_to : int >= 0, optional
        Input file coronal slice end. The default is 130.
    z_range_from : int >= 0, optional
        Input file axial slice start. The default is 15.
    z_range_to : int >= 0, optional
        Input file axial slice end. The default is 175.
    flavor : str, optional
        The input file name suffix. Defines which data to load, e.g. 'N4cor_WarpedSegmentationPosteriors2'.
    minmaxscaling : bool, optional
        Apply scaling to adjust minimum and maximum intensity to the range 0 and 1. The default is True.
    debug : bool, optional
        Display additional information for participant's file and spreadsheet matching. The default is False.

    Returns
    -------
    dict
        A dictionary containing the images as numpy array, the binary labels, the original labels, the covariates.

    """

    # Import data from Excel sheet
    df = pd.read_excel('/mnt/ssd1/dyrba/hippocampus_volume_relevance_DELCODE.xlsx', sheet_name='DELCODE_LRP_CMP')
    #print(df)
    sid = df['SID']
    grp = df['prmdiag'].replace(5, 3).replace(0, 1) # 5-AD, 2-MCI, 0-HC -> 3-AD, 1-HC
    age = df['age']
    sex = df['sex_bin_1female']
    #tiv = df['TIV_CAT12']
    field = df['FieldStrength']
    #amybin = df['ratio_Abeta42_40_pos']
    grpbin = (grp > 1) # 1=CN, ...
    
    
    # Scan for nifti file names
    dataFiles = sorted(glob.glob('/mnt/ssd1/dyrba/DELCODE_t1linear/*'+flavor+'.nii.gz'))
    numfiles = len(dataFiles)
    print('Found ', str(numfiles), ' nifti files')
    
    
    # Match covariate information
    cov_idx = [-1] * numfiles # list; array: np.full((numfiles, 1), -1, dtype=int)
    print('Matching covariates for loaded files ...')
    for i,id in enumerate(sid):
      p = [j for j,x in enumerate(dataFiles) if re.search('%s' % id, x)] # extract ID numbers from filename, translate to Excel row index
      if len(p)==0:
        if debug: print('Did not find %04d' % id) # did not find Excel sheet subject ID in loaded file selection
      else:
        if debug: print('Found %04d in %s: %s' % (id, p[0], dataFiles[p[0]]))
        cov_idx[p[0]] = i # store Excel index i for data file index p[0]
    print('Checking for scans not found in Excel sheet: ', sum(x<0 for x in cov_idx))
    
    print('Removing images without match')
    dataFiles = np.asarray(dataFiles)[np.asarray(cov_idx)>=0].tolist()
    cov_idx = np.asarray(cov_idx)[np.asarray(cov_idx)>=0].tolist()
    numfiles = len(dataFiles)
    
    labels = pd.DataFrame({'Group':grpbin}).iloc[cov_idx, :]
    grps = pd.DataFrame({'Group':grp, 'RID':util.add_marker('DELCODE_', sid)}).iloc[cov_idx, :]
    covariates = pd.DataFrame({'Age':age/100, 'Sex':sex, 'FS':(field-1.5)/1.5}).iloc[cov_idx, :]
    print("Covariates data frame size : ", covariates.shape)
    print(covariates.head())
    covariates = covariates.to_numpy(dtype=np.float32) # convert dataframe to nparray with 32bit types
    
    
    # Actually load nifti files into array
    print('Loading files...')
    # 1. dimension: subject
    # 2. dimension: img row
    # 3. dimension: img col
    # 4. dimension: img depth
    # 5. dimension: img channel
    images = np.zeros((numfiles, z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from, 1),
                      dtype=np.float32) # numfiles× z × x × y ×1; avoid 64bit types
    for i in range(numfiles):   # for loop over files and load
        if (i % 50 == 0):
            print('Loading file %d of %d' % (i+1, numfiles))
        images[i, :,:,:, 0] = read_nifti_data(dataFiles[i], x_range_from, x_range_to, y_range_from, y_range_to, z_range_from, z_range_to, minmaxscaling, aug)
    print('Successfully loaded files')
    print('Image array size: ', images.shape)
    
    return {'images': images, 'labels': labels, 'groups': grps, 'covariates': covariates}



#%%
def load_EDSD_data(x_range_from = 15, x_range_to = 175, y_range_from = 90, y_range_to = 130, z_range_from = 15, z_range_to = 175, flavor = 'N4cor_WarpedSegmentationPosteriors2', minmaxscaling = True, debug = False, aug=False):
    """
    Load the EDSD dataset and return it as numpy array.
    
    Typically, we will use a reduced FOV to save required memory size.
    
    # Clinica range for cropped image:
    x_range_from = 12; x_range_to = 181  # R - L
    y_range_from = 13; y_range_to = 221  # P - A
    z_range_from = 0; z_range_to = 179   # I - S
    
    :return: 

    Parameters
    ----------
    x_range_from : int >= 0, optional
        Input file saggital slice start. The default is 15.
    x_range_to : int >= 0, optional
        Input file saggital slice end. The default is 175.
    y_range_from : int >= 0, optional
        Input file coronal slice start. The default is 90.
    y_range_to : int >= 0, optional
        Input file coronal slice end. The default is 130.
    z_range_from : int >= 0, optional
        Input file axial slice start. The default is 15.
    z_range_to : int >= 0, optional
        Input file axial slice end. The default is 175.
    flavor : str, optional
        The input file name suffix. Defines which data to load, e.g. 'N4cor_WarpedSegmentationPosteriors2'.
    minmaxscaling : bool, optional
        Apply scaling to adjust minimum and maximum intensity to the range 0 and 1. The default is True.
    debug : bool, optional
        Display additional information for participant's file and spreadsheet matching. The default is False.

    Returns
    -------
    dict
        A dictionary containing the images as numpy array, the binary labels, the original labels, the covariates.

    """

    # Import data from Excel sheet
    df = pd.read_excel('/mnt/ssd1/dyrba/2016_Mar_14_Multicentre_DTI_aMCI_all.xlsx', sheet_name='Kopie_2014_May_06_Multicentre_D')
    sid = df['Multicenter_ID']
    grp = df['status'].replace("HC", 1).replace("aMCI", 2).replace("MCI", 2).replace("AD", 3)
    age = df['age']
    sex = df['gender'].replace("female", 1).replace("male", 0)
    #tiv = df['Total'] # TIV
    field = df['mri_scanner']
    grpbin = (grp > 1) # 1=CN, ...
        
    
    # Scan for nifti file names
    dataFiles = sorted(glob.glob('/mnt/ssd1/dyrba/EDSD_t1linear/*_'+flavor+'.nii.gz'))
    numfiles = len(dataFiles)
    print('Found ', str(numfiles), ' nifti files')
    
    
    # Match covariate information
    cov_idx = [-1] * numfiles # list; array: np.full((numfiles, 1), -1, dtype=int)
    print('Matching covariates for loaded files ...')
    for i,id in enumerate(sid):
      p = [j for j,x in enumerate(dataFiles) if re.search('%s_' % id, x)] # extract ID numbers from filename, translate to Excel row index
      if len(p)==0:
        if debug: print('Did not find %s' % id) # did not find Excel sheet subject ID in loaded file selection
      else:
        if debug: print('Found %s in %s: %s' % (id, p[0], dataFiles[p[0]]))
        cov_idx[p[0]] = i # store Excel index i for data file index p[0]
    print('Checking for scans not found in Excel sheet: ', sum(x<0 for x in cov_idx))
    
    labels = pd.DataFrame({'Group':grpbin}).iloc[cov_idx, :]
    grps = pd.DataFrame({'Group':grp, 'RID':util.add_marker('EDSD_', sid)}).iloc[cov_idx, :]
    covariates = pd.DataFrame({'Age':age/100, 'Sex':sex, 'FS':(field-1.5)/1.5}).iloc[cov_idx, :]
    print("Covariates data frame size : ", covariates.shape)
    print(covariates.head())
    covariates = covariates.to_numpy(dtype=np.float32) # convert dataframe to nparray with 32bit types
    
    
    # Actually load nifti files into array
    print('Loading files...')
    # 1. dimension: subject
    # 2. dimension: img row
    # 3. dimension: img col
    # 4. dimension: img depth
    # 5. dimension: img channel


    images = np.zeros((numfiles, z_range_to-z_range_from, x_range_to-x_range_from, y_range_to-y_range_from, 1),
                      dtype=np.float32) # numfiles× z × x × y ×1; avoid 64bit types
    for i in range(numfiles):   # for loop over files and load
        if (i % 50 == 0):
            print('Loading file %d of %d' % (i+1, numfiles))
        images[i, :,:,:, 0] = read_nifti_data(dataFiles[i], x_range_from, x_range_to, y_range_from, y_range_to, z_range_from, z_range_to, minmaxscaling,aug)
    print('Successfully loaded files')
    print('Image array size: ', images.shape)

    return {'images': images, 'labels': labels, 'groups': grps, 'covariates': covariates}


#%%
def read_nifti_data(filename, x_range_from, x_range_to, y_range_from, y_range_to, z_range_from, z_range_to, minmaxscaling, aug=False):
    """
    Reads the nifti file and loads the image data within the given field of view.

    Parameters
    ----------
    filename : string
        The input nifti file name.
    x_range_from : int >= 0
        Input file saggital slice start.
    x_range_to : int >= 0
        Input file saggital slice end.
    y_range_from : int >= 0
        Input file coronal slice start.
    y_range_to : int >= 0,
        Input file coronal slice end.
    z_range_from : int >= 0
        Input file axial slice start.
    z_range_to : int >= 0
        Input file axial slice end.
    minmaxscaling : bool
        Apply scaling to adjust minimum and maximum intensity to the range 0 and 1.

    Returns
    -------
    img : numpy array
        The loaded image data.

    """
    img = nib.load(filename)
    img = img.get_fdata()[x_range_from:x_range_to, y_range_from:y_range_to, z_range_from:z_range_to]
    img = np.transpose(img, (2, 0, 1)) # reorder dimensions to match coronal view z*x*y in MRIcron etc.
    img = np.flip(img) # flip all positions
    img = np.nan_to_num(img) # remove nan or inf values
    if minmaxscaling:
        img = (img - img.min()) / (img.max() - img.min()) # min/max scaling
    
    if aug:
        if np.random.choice([0,1,1]):   #would assure atleast 33% data is unagumented/pure.
            img = util.data_aug_inplace(img)
        
    return img


#%%
def quick_plot(dataset, index=0, jump=10):
    """
    Plots the 3D image of a given dataset and selected participant index as coronal 2D images.

    Parameters
    ----------
    dataset : dict
        Dataset dictionary as created by load_*_data..
    index : int, optional
        Participant index. The default is 0.
    jump : int>0, optional
        Display only each n'th slice. The default is 10.

    Returns
    -------
    None.

    """
    test_img = dataset['images'][index, :,:,:, 0]
    
    ma = np.max(test_img)
    mi = np.min(test_img)
    test_img = (test_img - mi) / (ma - mi) # normalising to (0-1) range
    
    print('displaying image ', dataset['groups'].iloc[index,:])
    for i in range(test_img.shape[2]):
        if (i % jump == 0): # only display each nth slice
            print('displaying slice: ', i)
            plt.figure()
            a = test_img[:,:,i]
            plt.imshow(a, cmap='gray')
