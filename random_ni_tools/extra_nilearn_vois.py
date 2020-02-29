from nilearn._utils.niimg_conversions import check_niimg_4d, check_niimg_3d
from nilearn.image import new_img_like
import sklearn
from sklearn import neighbors
from nilearn.image.resampling import coord_transform
import numpy as np

def create_spherical_vois(ref_img, seeds, radius=1):
    """ 
    Utility to create spherical masks for voi extraction.
    Careful, differing from NiftiSpheresMasker this function
    does not check whether vois are overlapping. The function
    puts out single masks for each coordinate. These can then
    be used with e.g. the NiftiMasker. Importantly note, that 
    this function is just a "remix" of nilearn's
    NiftiSpheresMasker! So check it out!
    Parameters
    ----------
    ref_img : 3D nifti image, which serves as the reference
        for VOI creation. Output masks will have the same 
        dimensions and affine as ref_img.
    seeds : a list of lists or tuples of size 3, defining
        the seed coordinates in MNI-space around which the 
        vois will be centered.
    radius : the radius of the VOI in mm, defaults to 1.
    Returns
    -------
    voi_masks : A list of length(seeds), containing
        a spherical mask for each seed in the dimensions
        of the ref_img.
    """

    niimg = check_niimg_3d(ref_img)
    affine = niimg.affine
    img_shape = niimg.shape

    X = niimg.get_data().reshape([-1, 1]).T

    mask_coords = list(np.ndindex(img_shape[:3]))

    voi_masks = []
    nearests = []

    for sx, sy, sz in seeds:
        nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        try:
            nearests.append(mask_coords.index(nearest))
        except ValueError:
            nearests.append(None)


    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = coord_transform(mask_coords[0], mask_coords[1],
                                  mask_coords[2], affine)
    mask_coords = np.asarray(mask_coords).T

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()

    clf.fit(mask_coords)

    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue
        A[i, nearest] = True

        voi_mask = new_img_like(niimg, 
                                      A[i].toarray().reshape(img_shape[:3]))

        voi_masks.append(voi_mask)
    
    return voi_masks