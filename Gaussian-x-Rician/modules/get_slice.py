#Name: Rafael Henrique
#e-mail: rafaelhenri@usp.br

"""
    Python code for getting slice from a nifti volume
"""

import numpy as np
import SimpleITK as sitk
import os


def get_axial_Slice_from_Nifti(path_to_volume,coord):
    """
    Routine to get axial slice from Nifti volume.

    Parameters
    ----------
    path_to_volume: String, path to nifti volume

    coord: Integer, axial coordinate

    Returns
    -------
    axial_slice: 2D Numpy Array 

    """

    img = sitk.ReadImage(os.path.dirname(__file__)+path_to_volume)
    img_array = sitk.GetArrayFromImage(img)

    axial_slice = img_array[coord,:,:]


    return np.asarray(axial_slice)
