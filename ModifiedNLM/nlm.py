import numpy as np
import sys

sys.path.append('./NLM_build')

#import NLM modified

from nlm_means import denoise_nl_means as gaussian_nlm
from estimate import estimate_sigma
from float_convert import img_as_float

def nlm_denoised(img):

    sigma_est = np.mean(estimate_sigma(img))

    img_float = img_as_float(img)

    denoised = gaussian_nlm(img_float, h=1.5*sigma_est,fast_mode=False,
            patch_size=5,patch_distance=6,multichannel=False,preserve_range=True)
    return denoised
