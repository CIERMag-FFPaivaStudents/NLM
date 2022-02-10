
import scipy.stats
import numpy as np
import pywt

def _sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """Calculate the robust median estimator of the noise standard deviation.
    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.
    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).
    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently "
                         "supported")
    return sigma


def estimate_sigma(image, average_sigmas=False, multichannel=False, *,
                   channel_axis=None):
    """
    Robust wavelet-based estimator of the (Gaussian) noise standard deviation.
    Parameters
    ----------
    image : ndarray
        Image for which to estimate the noise standard deviation.
    average_sigmas : bool, optional
        If true, average the channel estimates of `sigma`.  Otherwise return
        a list of sigmas corresponding to each channel.
    multichannel : bool
        Estimate sigma separately for each channel. This argument is
        deprecated: specify `channel_axis` instead.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.
        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    Returns
    -------
    sigma : float or list
        Estimated noise standard deviation(s).  If `multichannel` is True and
        `average_sigmas` is False, a separate noise estimate for each channel
        is returned.  Otherwise, the average of the individual channel
        estimates is returned.
    Notes
    -----
    This function assumes the noise follows a Gaussian distribution. The
    estimation algorithm is based on the median absolute deviation of the
    wavelet detail coefficients as described in section 4.2 of [1]_.
    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    Examples
    --------
    >>> import skimage.data
    >>> from skimage import img_as_float
    >>> img = img_as_float(skimage.data.camera())
    >>> sigma = 0.1
    >>> rng = np.random.default_rng()
    >>> img = img + sigma * rng.standard_normal(img.shape)
    >>> sigma_hat = estimate_sigma(img, channel_axis=None)
    """
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        _at = functools.partial(utils.slice_at_axis, axis=channel_axis)
        nchannels = image.shape[channel_axis]
        sigmas = [estimate_sigma(
            image[_at(c)], channel_axis=None) for c in range(nchannels)]
        if average_sigmas:
            sigmas = np.mean(sigmas)
        return sigmas
    elif image.shape[-1] <= 4:
        msg = f'image is size {image.shape[-1]} on the last axis, '\
              f'but channel_axis is None. If this is a color image, '\
              f'please set channel_axis=-1 for proper noise estimation.'
        warn(msg)
    coeffs = pywt.dwtn(image, wavelet='db2')
    detail_coeffs = coeffs['d' * image.ndim]
    return _sigma_est_dwt(detail_coeffs, distribution='Gaussian')
