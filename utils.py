from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import pandas as pd
import pydicom
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
import nibabel as nib
from glob import glob
from scipy import ndimage
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

# Source: https://www.kaggle.com/kmader/dsb-lung-segmentation-algorithm/notebook
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage, os
from skimage.morphology import ball, disk, dilation, binary_erosion, remove_small_objects, erosion, closing, reconstruction, binary_closing
from skimage.measure import label,regionprops, perimeter
from skimage.morphology import binary_dilation, binary_opening
from skimage.filters import roberts, sobel
from skimage import measure, feature
from skimage.segmentation import clear_border, mark_boundaries
from skimage import data
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.misc
from glob import glob
from skimage.io import imread
def get_segmented_lungs(raw_im, plot=False):
    '''
    Original function changes input image (ick!)
    '''
    im = raw_im.copy()
    
    '''
    This funtion segments the lungs from the given 2D slice.
    '''
    if plot == True:
        f, plots = plt.subplots(8, 1, figsize=(5, 40))
    '''
    Step 1: Convert into a binary image. 
    '''
    binary = im < -400
    if plot == True:
        plots[0].axis('off')
        plots[0].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 2: Remove the blobs connected to the border of the image.
    '''
    cleared = clear_border(binary)
    if plot == True:
        plots[1].axis('off')
        plots[1].imshow(cleared, cmap=plt.cm.bone) 
    '''
    Step 3: Label the image.
    '''
    label_image = label(cleared)
    if plot == True:
        plots[2].axis('off')
        plots[2].imshow(label_image, cmap=plt.cm.bone) 
    '''
    Step 4: Keep the labels with 2 largest areas.
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:                
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[3].axis('off')
        plots[3].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 5: Erosion operation with a disk of radius 2. This operation is 
    seperate the lung nodules attached to the blood vessels.
    '''
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    if plot == True:
        plots[4].axis('off')
        plots[4].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 6: Closure operation with a disk of radius 10. This operation is 
    to keep nodules attached to the lung wall.
    '''
    selem = disk(10)
    binary = binary_closing(binary, selem)
    if plot == True:
        plots[5].axis('off')
        plots[5].imshow(binary, cmap=plt.cm.bone) 
    '''
    Step 7: Fill in the small holes inside the binary mask of lungs.
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[6].axis('off')
        plots[6].imshow(binary, cmap=plt.cm.bone) 
        
    return binary

def display_ct_volume(vol, cols=15, title=None, cf=1., rf=1.):
    rows = vol.shape[-1] // cols + 1
    plt.figure(figsize=(max(int(cols*cf), 1), max(int(rows*rf), 1)))
    plt.subplots_adjust(wspace=0.01, hspace=0.1)
    if title is not None:
        print(title)
    for i in range(vol.shape[-1]):
        plt.subplot(rows, cols, i+1)
        plt.imshow(vol[:, :, i], cmap="gray", vmin=0, vmax=1)
        plt.axis('off')
    plt.show()
    
def chest_ct_visualization(view_ct=True, view_xray=True, view_fxray=True, 
                           data_loader=None, config=None, cols=32,
                          ):
    assert data_loader is not None
    if data_loader == 'luna16':
        vol = luna16_data_loader(config)
    elif data_loader == 'kits19':
        vol = kits19_data_loader(config)
    elif data_loader == 'lits17':
        vol = lits17_data_loader(config)
    elif data_loader == 'dsb17':
        vol = dsb17_data_loader(config)
    elif data_loader == 'ctpa':
        vol = ctpa_data_loader(config)
    elif data_loader == 'lndb19':
        vol = lndb19_data_loader(config)
    elif data_loader == 'rsnastr20':
        vol = rsnastr20_data_loader(config)
    else:
        raise
    print('{}_data_loader vol: {} | {:.2f} ~ {:.2f}'.format(data_loader, vol.shape, np.min(vol), np.max(vol)))
    
    if view_ct:
        display_ct_volume(vol, title=data_loader)
    if view_xray:
        fake_xray = np.mean(vol, axis=-1)
        plt.imshow(fake_xray, cmap="gray", vmin=0, vmax=1); plt.axis('off'); plt.show()
    subvol = generate_subvolume(config, vol)
    print('subvol: {} | {:.2f} ~ {:.2f}'.format(subvol.shape, np.min(subvol), np.max(subvol)))
    for i in range(subvol.shape[0]):
        if view_fxray:
            fsubvol = np.mean(subvol[i], axis=-1)
            plt.imshow(fsubvol, cmap="gray", vmin=0, vmax=1); plt.axis('off'); plt.show()
        display_ct_volume(subvol[i], cols=cols)

def bernstein_poly(i, n, t):
    '''
     The Bernstein polynomial of n, i as a function of t
    '''

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    '''
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    '''

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[:, 
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y, 
                                                           noise_z:noise_z+block_noise_size_z]
        cnt -= 1
    return x

def vol_orientation(vol):

    fake_xray = np.mean(vol, axis=-1)
    upper = np.sum(fake_xray[:fake_xray.shape[0]//4, :]>0.5)
    lower = np.sum(fake_xray[3*fake_xray.shape[0]//4:, :]>0.5)
    if upper > lower:
        vol = vol[::-1,...]
    
    return vol

def luna16_data_loader(config, index_subset=[i for i in range(10)]):
    # Return
    # vol: (512, 512, 133) | 0 ~ 1

    luna_subset_path = os.path.join( config.luna16_data, 'subset'+str(random.choice(index_subset)) )
    file_list = glob(os.path.join(luna_subset_path, '*.mhd'))
    img_file = random.choice(file_list)
    
    itk_img = sitk.ReadImage(img_file) 
    vol = sitk.GetArrayFromImage(itk_img)
    vol = vol.transpose(2, 1, 0)

    vol = np.array(vol)
    hu_min, hu_max = -1000, 1000
    vol[vol < hu_min] = hu_min
    vol[vol > hu_max] = hu_max
    vol = 1.0*(vol-hu_min) / (hu_max-hu_min)
    
    vol = vol[...,::-1]
    vol = np.transpose(vol, (1, 0, 2))
            
    return vol

def lndb19_data_loader(config, index_subset=[i for i in range(6)]):
    # Return
    # vol: (512, 512, 133) | 0 ~ 1

    lndb_subset_path = os.path.join( config.lndb19_data, 'data'+str(random.choice(index_subset)) )
    file_list = glob(os.path.join(lndb_subset_path, '*.mhd'))
    img_file = random.choice(file_list)
    
    itk_img = sitk.ReadImage(img_file) 
    vol = sitk.GetArrayFromImage(itk_img)
    vol = vol.transpose(2, 1, 0)

    vol = np.array(vol)
    hu_min, hu_max = -1000, 1000
    vol[vol < hu_min] = hu_min
    vol[vol > hu_max] = hu_max
    vol = 1.0*(vol-hu_min) / (hu_max-hu_min)
    
    vol = np.transpose(vol, (1, 0, 2))
            
    return vol

def dicom_reader(dicom_name):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_name)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    return image

def sitk_to_numpy(sitkImage):

    return sitk.GetArrayFromImage(sitkImage)[::-1,:,:]

def ctpa_data_loader(config):
    # Return
    # vol: (512, 512, 133) | 0 ~ 1

    all_files = os.listdir( config.ctpa_data )
    file_name = random.choice( all_files )
    dicom_path = os.path.join(config.ctpa_data, file_name)
    vol = sitk_to_numpy(dicom_reader(dicom_path))
    vol = vol.transpose(2, 1, 0)

    vol = np.array(vol)
    hu_min, hu_max = -1000, 1000
    vol[vol < hu_min] = hu_min
    vol[vol > hu_max] = hu_max
    vol = 1.0*(vol-hu_min) / (hu_max-hu_min)
    
    vol = np.transpose(vol, (1, 0, 2))

    return vol

def dsb17_data_loader(config):
    # Return
    # vol: (512, 512, 133) | 0 ~ 1

    all_files = os.listdir( config.dsb17_data )
    file_name = random.choice( all_files )
    dicom_path = os.path.join(config.dsb17_data, file_name)
    vol = sitk_to_numpy(dicom_reader(dicom_path))
    vol = vol.transpose(2, 1, 0)

    vol = np.array(vol)
    hu_min, hu_max = -1000, 1000
    vol[vol < hu_min] = hu_min
    vol[vol > hu_max] = hu_max
    vol = 1.0*(vol-hu_min) / (hu_max-hu_min)
    
    vol = np.transpose(vol, (1, 0, 2))

    return vol

def lits17_data_loader(config):
    # Return
    # vol: (512, 512, 133) | 0 ~ 1

    all_files = os.listdir( config.lits17_data )
    all_files = [files for files in all_files if 'volume-' in files]
    file_name = random.choice( all_files )
    vol = nib.load(os.path.join(config.lits17_data, file_name))
    vol = vol.get_fdata()

    vol = np.array(vol)
    hu_min, hu_max = -1000, 1000
    vol[vol < hu_min] = hu_min
    vol[vol > hu_max] = hu_max
    vol = 1.0*(vol-hu_min) / (hu_max-hu_min)
    
    vol = vol[...,::-1]
    vol = np.transpose(vol, (1, 0, 2))
    
    vol = vol_orientation(vol)

    return vol

def kits19_data_loader(config):
    # Return
    # vol: (512, 512, 133) | 0 ~ 1

    all_files = os.listdir( config.kits19_data )
    all_files = [files for files in all_files if 'case_' in files]
    file_name = random.choice( all_files )
    vol = nib.load(os.path.join(config.kits19_data, file_name, 'imaging.nii.gz'))
    vol = vol.get_fdata()
    vol = vol.transpose(2, 1, 0)

    vol = np.array(vol)
    hu_min, hu_max = -1000, 1000
    vol[vol < hu_min] = hu_min
    vol[vol > hu_max] = hu_max
    vol = 1.0*(vol-hu_min) / (hu_max-hu_min)
    
    vol = np.transpose(vol, (1, 0, 2))

    return vol

def transform_to_hu(slices):
    images = np.stack([file.pixel_array for file in slices])
    images = images.astype(np.int16)

    # convert to HU
    for n in range(len(slices)):
        
        intercept = slices[n].RescaleIntercept
        slope = slices[n].RescaleSlope
        
        if slope != 1:
            images[n] = slope * images[n].astype(np.float64)
            images[n] = images[n].astype(np.int16)
            
        images[n] += np.int16(intercept)
    
    return np.array(images, dtype=np.int16)

def rsnastr20_data_loader(config):
    # Return
    # vol: (512, 512, 133) | 0 ~ 1

    train = pd.read_csv(os.path.join(config.rsnastr20_data, 'train.csv'))
    test = pd.read_csv(os.path.join(config.rsnastr20_data, 'test.csv'))
    
    num_case = train.shape[0] + test.shape[0]
    index = random.randint(0, num_case - 1)
    
    if index < train.shape[0]:
        dcm_path = os.path.join(config.rsnastr20_data, 
                                'train', 
                                train.StudyInstanceUID[index], 
                                train.SeriesInstanceUID[index],
                               )
    else:
        dcm_path = os.path.join(config.rsnastr20_data, 
                                'test', 
                                test.StudyInstanceUID[index - train.shape[0]], 
                                test.SeriesInstanceUID[index - train.shape[0]],
                               )
    
    slices = [pydicom.dcmread(os.path.join(dcm_path, file)) for file in os.listdir(dcm_path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    
    vol = transform_to_hu(slices)
    
    hu_min, hu_max = -1000, 1000
    vol[vol < hu_min] = hu_min
    vol[vol > hu_max] = hu_max
    vol = 1.0*(vol-hu_min) / (hu_max-hu_min)
    
    vol = np.einsum('kij->ijk',vol)
    
    vol = vol[...,::-1]
    
    return vol

def useful_subvol(config, vol):
    # Input:
    # vol: (64, 64, 64+3) | 0.0 ~ 1.0
    # Return
    # True/False
    d_img = np.zeros((config.input_rows, config.input_cols, config.input_deps), dtype=float)
    
    for d in range(config.input_deps):
        for i in range(config.input_rows):
            for j in range(config.input_cols):
                for k in range(config.len_vision):
                    if vol[i, j, d+k] >= config.hu_thred:
                        d_img[i, j, d] = k
                        break
                    if k == config.len_vision-1:
                        d_img[i, j, d] = k
                        
    d_img = d_img.astype('float32')
    d_img /= (config.len_vision - 1)
    d_img = 1.0 - d_img
    
    if np.sum(d_img) > config.lung_max * config.input_rows * config.input_cols * config.input_deps:
        # print('not useful {}'.format(Integral_Image_Filters(vol)))
        return False
    else:
        # print('useful {}'.format(Integral_Image_Filters(vol)))
        return True
    
def choose_from_lung(lung_coordinate, 
                     rows, cols, deps,
                     input_rows, input_cols, input_deps,
                     border_px,
                    ):
    get_coordinate = random.randint(0, lung_coordinate[0].shape[0]-1)
    cx = lung_coordinate[0][get_coordinate]
    cy = lung_coordinate[1][get_coordinate]
    cz = lung_coordinate[2][get_coordinate]

    cx = min(cx, rows-input_rows//2-border_px)
    cx = max(cx, input_rows//2+border_px)

    cy = min(cy, cols-input_cols//2-border_px)
    cy = max(cy, input_cols//2+border_px)

    cz = min(cz, deps-input_deps//2-border_px)
    cz = max(cz, input_deps//2+border_px)
    
    return cx, cy, cz

def choose_from_volume(rows, cols, deps,
                       input_rows, input_cols, input_deps,
                       border_px,
                      ):
    
    cx = random.randint(input_rows//2+border_px*4, rows-input_rows//2-border_px*4)
    cy = random.randint(input_cols//2+border_px*4, cols-input_cols//2-border_px*4)
    cz = random.randint(input_deps//2+border_px, deps-input_deps//2-border_px)
    
    return cx, cy, cz

def generate_subvolume(config, vol, within_lung=0.75):
    # Input:
    # vol: (512, 512, 250) | 0.0 ~ 1.0
    # Return
    # subvol: (32, 64, 64, 64) | 0.0 ~ 1.0

    border_px = 10
    rows, cols, deps = vol.shape
    
    subvol = np.zeros((config.num_subvol_per_patient, 
                       config.input_rows, 
                       config.input_cols, 
                       config.input_deps), dtype=float)

    ''' Find lung area
    lung_mask = np.zeros((vol.shape), dtype='int')
    for z in range(config.input_deps//2+border_px, deps-config.input_deps//2-border_px, 1):
        lung_mask[:,:,z] = get_segmented_lungs(vol[:,:,z]*2000.0-1000)
        
    if np.sum(lung_mask) < 100:
        return None
    
    lung_coordinate = np.where(lung_mask == True)
    '''
    
    num_subvol = 0
    cnt = 0
    while True:
        cnt += 1
        if cnt > config.num_subvol_per_patient * 10:
            return None
        
        ''' Extract from lung
        if random.random() < within_lung:
            cx, cy, cz = choose_from_lung(lung_coordinate, 
                                          rows, cols, deps,
                                          config.input_rows, config.input_cols, config.input_deps,
                                          border_px,
                                         )
        else:
            cx, cy, cz = choose_from_volume(rows, cols, deps,
                                            config.input_rows, config.input_cols, config.input_deps,
                                            border_px,
                                           )
        '''
        crop_rows = random.randint(config.crop_rows_min, min(rows - 8*border_px, config.crop_rows_max))
        crop_cols = random.randint(config.crop_cols_min, min(cols - 8*border_px, config.crop_cols_max))
        crop_deps = random.randint(config.crop_deps_min, min(deps - 2*border_px, config.crop_deps_max))
        
        
        cx, cy, cz = choose_from_volume(rows, cols, deps,
                                        crop_rows, crop_cols, crop_deps,
                                        border_px,
                                       ) 
        crop_vol = vol[cx-crop_rows//2:cx+crop_rows//2,
                       cy-crop_cols//2:cy+crop_cols//2,
                       cz-crop_deps//2:cz+crop_deps//2,
                      ]
        crop_vol = resize(crop_vol, (config.input_rows,config.input_cols,config.input_deps), anti_aliasing=True)
        subvol[num_subvol] = crop_vol
        num_subvol += 1
        cnt = 0

        if num_subvol == config.num_subvol_per_patient:
            break
    
            
    return np.array(subvol)

def collect_subvol(config):
    img = []
    variety = copy.deepcopy(config.variety)
    while variety > 0:
        dataset_name = random.choice( config.datasets )

        if dataset_name == 'luna16':
            vol = luna16_data_loader(config) # vol: (512, 512, 133) | 0 ~ 1

        elif dataset_name == 'ctpa':
            vol = ctpa_data_loader(config) # vol: (512, 512, 133) | 0 ~ 1
            
        elif dataset_name == 'dsb17':
            vol = dsb17_data_loader(config) # vol: (512, 512, 133) | 0 ~ 1
            
        elif dataset_name == 'lits17':
            vol = lits17_data_loader(config) # vol: (512, 512, 133) | 0 ~ 1
            
        elif dataset_name == 'lndb19':
            vol = lndb19_data_loader(config) # vol: (512, 512, 133) | 0 ~ 1
            
        elif dataset_name == 'kits19':
            vol = kits19_data_loader(config) # vol: (512, 512, 133) | 0 ~ 1
        
        elif dataset_name == 'rsnastr20':
            vol = rsnastr20_data_loader(config) # vol: (512, 512, 133) | 0 ~ 1

        else:
            raise
        
        subvol = generate_subvolume(config, vol) # subvol: (32, 64, 64, 64) | 0.0 ~ 1.0
        if subvol is not None:
            img.extend(np.expand_dims(subvol, axis=1))
            variety -= 1

    return np.array(img)

def generate_pair(config, status=None):
    assert status is not None

    while True:
        img = collect_subvol(config=config)
        # print('img: {} | {} ~ {}'.format(img.shape, np.min(img), np.max(img)))

        index = [i for i in range(img.shape[0])]
        random.shuffle(index)
        y = img[index[:config.batch_size]]
        x = copy.deepcopy(y)
        for n in range(config.batch_size):
            
            # Autoencoder
            x[n] = copy.deepcopy(y[n])
            
            # Flip
            x[n], y[n] = data_augmentation(x[n], y[n], config.flip_rate)

            # Local Shuffle Pixel
            x[n] = local_pixel_shuffling(x[n], prob=config.local_rate)
            
            # Apply non-Linear transformation with an assigned probability
            x[n] = nonlinear_transformation(x[n], config.nonlinear_rate)
            
            # Inpainting & Outpainting
            if random.random() < config.paint_rate:
                if random.random() < config.inpaint_rate:
                    # Inpainting
                    x[n] = image_in_painting(x[n])
                else:
                    # Outpainting
                    x[n] = image_out_painting(x[n])

        # Save sample images module
        if config.save_samples is not None and status == 'train' and random.random() < config.sample_png_rate:
            n_sample = random.choice( [i for i in range(config.batch_size)] )
            sample_1 = np.concatenate((x[n_sample,0,:,:,2*config.input_deps//6], y[n_sample,0,:,:,2*config.input_deps//6]), axis=1)
            sample_2 = np.concatenate((x[n_sample,0,:,:,3*config.input_deps//6], y[n_sample,0,:,:,3*config.input_deps//6]), axis=1)
            sample_3 = np.concatenate((x[n_sample,0,:,:,4*config.input_deps//6], y[n_sample,0,:,:,4*config.input_deps//6]), axis=1)
            sample_4 = np.concatenate((x[n_sample,0,:,:,5*config.input_deps//6], y[n_sample,0,:,:,5*config.input_deps//6]), axis=1)
            final_sample = np.concatenate((sample_1, sample_2, sample_3, sample_4), axis=0)
            final_sample = final_sample * 255.0
            final_sample = final_sample.astype(np.uint8)
            file_name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(10)])+'.'+config.save_samples
            imageio.imwrite(os.path.join(config.sample_path, config.exp_name, file_name), final_sample)

        yield (x, y)

if __name__ == '__main__':
    from config import models_genesis_config
    class set_args():
        data = None
        weights = None
    args = set_args()
    conf = models_genesis_config(args=args)
    conf.display()
    
    vol = luna16_data_loader(conf)
    print('luna16_data_loader vol: {} | {} ~ {}'.format(vol.shape, np.min(vol), np.max(vol)))

    vol = kits19_data_loader(conf)
    print('kits19_data_loader vol: {} | {} ~ {}'.format(vol.shape, np.min(vol), np.max(vol)))

    vol = lits17_data_loader(conf)
    print('lits17_data_loader vol: {} | {} ~ {}'.format(vol.shape, np.min(vol), np.max(vol)))

    vol = dsb17_data_loader(conf)
    print('dsb17_data_loader vol: {} | {} ~ {}'.format(vol.shape, np.min(vol), np.max(vol)))
    
    vol = ctpa_data_loader(conf)
    print('ctpa_data_loader vol: {} | {} ~ {}'.format(vol.shape, np.min(vol), np.max(vol)))
    
    vol = lndb19_data_loader(conf)
    print('lndb19_data_loader vol: {} | {} ~ {}'.format(vol.shape, np.min(vol), np.max(vol)))

#     x, y = generate_pair(conf, status='train')
#     print('x: {} | {} ~ {}'.format(x.shape, np.min(x), np.max(x)))
#     print('y: {} | {} ~ {}'.format(y.shape, np.min(y), np.max(y)))

    # vol = luna16_data_loader(conf)
    # print('vol: {} | {} ~ {}'.format(vol.shape, np.min(vol), np.max(vol)))

    # subvol = generate_subvolume(conf, vol)
    # print('subvol: {} | {} ~ {}'.format(subvol.shape, np.min(subvol), np.max(subvol)))
