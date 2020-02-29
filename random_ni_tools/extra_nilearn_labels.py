"""
A little function to extract the labels of some coordinates
according to some atlas, using nilearn. (and nilearn datasets etc.)
"""
import nilearn
import numpy as np
from nilearn.image import load_img, iter_img
from nilearn.image import resampling
from nilearn import datasets


def get_atlas_label(atl_map, atl_labels, atl_indices, coord):
    
    atl_label = atl_map.get_data()[coord[0], coord[1], coord[2]]

    label_idx =np.where(np.asarray(atl_indices, dtype =int)  == atl_label)[0]
    
    if len(label_idx) == 0: 
        
        
        atl_label = 'Not found'
        
        return atl_label
        
    atl_label = atl_labels[label_idx[0]]


    return atl_label    


def get_prob_atlas_label(prob_map, prob_labels, coord, thresh = None):
    
    label_prob = list()
    
    for slices in iter_img(prob_map):
        
        label_prob.append(slices.get_data()[coord[0], coord[1], coord[2]])
        
    # Get probability above a certain threshold or max:
    
    if thresh is None:
        thresh = np.max(label_prob)    
        
    if thresh == 0:
        thresh = 1
                
        
    
    label_idx = np.where(np.asarray(label_prob) >= thresh)[0]
    
    labels_out = list()
    proba_out = list()
    for idx in label_idx:
        
        proba_out.append(label_prob[idx])
        labels_out.append(prob_labels[idx])
        
    
    return labels_out, proba_out
    
    
def coordinate_label(mni_coord, atlas = 'aal', thresh = None, ret_proba = False):
    
    if atlas == 'aal':
        atl = datasets.fetch_atlas_aal()
        atl.prob = False
    elif atlas == 'harvard_oxford':
        atl = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
        atl.prob = True

    elif atlas == 'destrieux':
        atl = datasets.fetch_atlas_destrieux_2009()
        atl.indices = atl.labels['index']
        atl.labels = atl.labels['name']
        atl.prob = False
        
    atl_map = load_img(atl.maps)
    atl_aff = atl_map.affine

        
    if atl.prob == True:
        atl_labels = atl.labels
    if atl.prob == False:
        atl_labels = atl.labels
        atl_indices = atl.indices

    labels_out = list()    
    
    for coord in mni_coord: 
        
        mat_coord = np.asarray(resampling.coord_transform(coord[0], coord[1],
				coord[2], np.linalg.inv(atl_aff)), dtype = int)

        if atl.prob == True and ret_proba == True:
            
            lab_out = get_prob_atlas_label(atl_map, atl_labels, mat_coord, 
					   thresh = thresh)
        
        elif atl.prob == True and ret_proba == False:
            
            lab_out, _ = get_prob_atlas_label(atl_map, atl_labels, mat_coord, 
					      thresh = thresh)
        
        elif atl.prob == False:
            
            lab_out = get_atlas_label(atl_map, atl_labels, atl_indices, 
				      mat_coord)
        
        labels_out.append(lab_out)
    
    return labels_out
            