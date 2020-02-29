# Preface
This package contains a few utilities I'd like to have in nilearn. Maybe, someday, when I have the time, there might be a pull request to [https://nilearn.github.io/].
Please note especially that the extra_nilearn_vois are simply re-used code from [NiftiSpheresMasker][https://github.com/nilearn/nilearn/blob/master/nilearn/input_data/nifti_spheres_masker.py] of nilearn's input_data module. They deserve all the credit.

# Introduction
Currently there are only two main functions in here:
1. To create spherical vois in MNI space, which I sought might be helpful to work around the [https://github.com/nilearn/nilearn/issues/1192]
2. Labeling coordinates using atlasses which are available in nilearn. Main rational was to quickly create a list of labels for the labeling of e.g. connectivity matrices. I do not claim any accuracy, more as a simple tool to get a quicker overview.
3. Visualizing SPM DCM files using networkx

# TODOS:
Way to many, tests, documentation etc. etc. etc.

