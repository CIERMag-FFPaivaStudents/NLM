"""
    Rafael Henrique, e-mail: rafaelhenri@usp.br, March 2022

    Code for testing modifiedNLM package
"""

#external libraries needed
import numpy as np
import SimpleITK as sitk

from ..filter.modified_nl_means import rician_denoise_nl_means
from ..estimate.noise_estimate import rician_estimate

def test_pass():


    input_img_path="modifiedNLM/tests/samples/t1_icbm_normal_1mm_pn0_rf0_rician_slice_20.nii"

    img = sitk.ReadImage(input_img_path)

    img_vec = sitk.GetArrayFromImage(img)

    sigma_est = np.mean(rician_estimate(img_vec))

    denoised = rician_denoise_nl_means(img_vec, h=1.2*sigma_est,fast_mode=False,
            patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)


    output_img_path="modifiedNLM/tests/samples/denoised_t1_icbm_normal_1mm_pn0_rf0_rician_slice_20.nii"
    output_test_img = sitk.ReadImage(output_img_path)
    expected = sitk.GetArrayFromImage(output_test_img)

    assert np.array_equal(expected, denoised), "\n There is something wrong! \n"
