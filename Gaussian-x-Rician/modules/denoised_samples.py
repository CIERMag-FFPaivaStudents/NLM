#Name: Rafael Henrique
#e-mail: rafaelhenri@usp.br

"""
    Python code for generating image samples
"""

import numpy as np
import sys

sys.path.append('..')

import add_noises as add

#rician NLM modules
from modifiedNLM.filter.modified_nl_means import rician_denoise_nl_means
from modifiedNLM.estimate.noise_estimate import rician_estimate

#gaussian NLM modules, using skimage
from skimage.restoration import denoise_nl_means as gaussian_denoise_nl_means
from skimage.restoration import estimate_sigma as gaussian_estimate

#Generate image samples
def generate_denoised_samples_slices(slice, noise):
    """
    Routine to generate image samples.

    Parameters
    ----------
    slice: 2D Numpy Array

    noise: Integer number between 1 and 100

    Returns
    -------
    List of image samples

    """

    slice_with_gaussian_noise = add.gaussian_noise(slice,noise)

    slice_with_rician_noise = add.rician_noise(slice,noise)

    #Rician noise part
    img = slice_with_rician_noise

    sigma_est_rician = np.mean(rician_estimate(img))
    sigma_est_gaussian = np.mean(gaussian_estimate(img))

    denoised_rician_rician = rician_denoise_nl_means(img, h=1.2*sigma_est_rician,fast_mode=False,
            patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)

    denoised_rician_gaussian = gaussian_denoise_nl_means(img, h=1.2*sigma_est_rician,fast_mode=False,
            patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)

    denoised_gaussian_gaussian = gaussian_denoise_nl_means(img, h=1.2*sigma_est_gaussian,fast_mode=False,
            patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)

    denoised_gaussian_rician = rician_denoise_nl_means(img, h=1.2*sigma_est_gaussian,fast_mode=False,
            patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)

    RrR = denoised_rician_rician
    RrG = denoised_rician_gaussian
    RgG = denoised_gaussian_gaussian
    RgR = denoised_gaussian_rician


    #Gaussian noise part
    img = slice_with_gaussian_noise

    sigma_est_rician = np.mean(rician_estimate(img))
    sigma_est_gaussian = np.mean(gaussian_estimate(img))

    denoised_rician_rician = rician_denoise_nl_means(img, h=1.2*sigma_est_rician,fast_mode=False,
        patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)

    denoised_rician_gaussian = gaussian_denoise_nl_means(img, h=1.2*sigma_est_rician,fast_mode=False,
        patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)

    denoised_gaussian_gaussian = gaussian_denoise_nl_means(img, h=1.2*sigma_est_gaussian,fast_mode=False,
        patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)

    denoised_gaussian_rician = rician_denoise_nl_means(img, h=1.2*sigma_est_gaussian,fast_mode=False,
        patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)

    GrR = denoised_rician_rician
    GrG = denoised_rician_gaussian
    GgG = denoised_gaussian_gaussian
    GgR = denoised_gaussian_rician

    #XxX = NOISEestimativeFILTER

    return [RrR,RrG,RgG,RgR,GrR,GrG,GgG,GgR]
