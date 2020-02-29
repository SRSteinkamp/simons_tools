from nilearn._utils.testing import assert_raises_regex
from numpy.testing import assert_array_equal
from nilearn._utils.testing import generate_fake_fmri
import nibable as nb
import numpy np
from nilearn_extras import create_spherical_vois

def test_output_length():
    data = np.random.random((3, 3, 3))
    img = nb.Nifti1Image(data, np.eye(4))
    
    seeds = [(0, 0, 0), (1,1,1), (2,2,2)]
    vois = create_spherical_vois(img, seeds, radius=1)
    assert(len(seeds) == len(vois), 'Output length unequal!')
    
    seeds2 = [(0, 0, 0)]
    vois2 = create_spherical_vois(img, seeds, radius=1)
    assert(len(seeds2) == len(vois2), 'Output length unequal!')
    
def test_output_img():
    data = np.random.random((3, 3, 3))
    img = nb.Nifti1Image(data, np.eye(4))
    
    seeds = [(0, 0, 0), (1,1,1), (2,2,2)]
    vois = create_spherical_vois(img, seeds, radius=1)
    
    for vv in vois:
        assert_array_equal(vv.shape, img.shape)
        assert_array_equal(vv.affine, img.affine)
    
    seeds2 = [(0, 0, 0)]
    vois2 = create_spherical_vois(img, seeds, radius=1)
    
    for vv in vois:
        assert_array_equal(vv.shape, img.shape)
        assert_array_equal(vv.affine, img.affine)
   
def test_errors():
    seeds = ([1, 2])
    data = np.random.random((3, 3, 3))
    img = nb.Nifti1Image(data, np.eye(4))
    
    assert_raises_regex(TypeError, "int' object .+", create_spherical_vois, img, seeds)