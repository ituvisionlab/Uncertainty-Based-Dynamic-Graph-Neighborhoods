"""
Data loaders based on tensorpack
"""

import numpy as np
from gcn_pancreas.utilities import nparrays as arrtools


def get_pancreas_generator(sample_name):
    sample_vol_name =  sample_name[0]
    reference_vol_name = sample_name[1]

    volume = np.load(sample_vol_name)
    reference = np.load(reference_vol_name)
    reference[reference != 0] = 1
    y, x, z = volume.shape

    for i in range(z):
        vol_slice = volume[:, :, i]
        reference_slice = reference[:, :, i]
        vol_slice = arrtools.extend2_before(vol_slice)
        reference_slice = arrtools.extend2_before(reference_slice)
        yield[vol_slice, reference_slice]
