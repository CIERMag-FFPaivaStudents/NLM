#Name: Rafael Henrique
#e-mail: rafaelhenri@usp.br

"""
    Code for testing modifiedNLM package
"""

#external libraries needed
import numpy as np
import SimpleITK as sitk
import os

from ..filter.modified_nl_means import rician_denoise_nl_means
from ..estimate.noise_estimate import rician_estimate

def test_pass():
    """
    Testing code for package modifiedNLM, if assert is not TRUE there is something wrong with
    package installation or dependencies, therefore check out READ.md file.

    This code compares an expected denoised image (We know this image was correctly denoised) with a denoised image generated
    in execution time, using the same input image as expected denoised image. If both are equal, therefore modifiedNLM was correctly
    installed.
    """

    input_img_path="/samples/input_image_test.nii"

    img = sitk.ReadImage(os.path.dirname(__file__)+input_img_path)

    img_array = sitk.GetArrayFromImage(img)

    sigma_est = np.mean(rician_estimate(img_array))

    denoised_array = rician_denoise_nl_means(img_array, h=1.2*sigma_est,fast_mode=False,
            patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)


    expected_img_path="/samples/expected_image_test.nii"
    expected_test_img = sitk.ReadImage(os.path.dirname(__file__)+expected_img_path)
    expected_array = sitk.GetArrayFromImage(expected_test_img)

    assert np.array_equal(expected_array, denoised_array), "\n There is something wrong! \n"
