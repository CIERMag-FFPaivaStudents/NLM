#Name: Rafael Henrique
#e-mail: rafaelhenri@usp.br

"""
    Python code for calculate quality metrics
"""

import numpy as np

def SSIM(I,I_hat,mask):
    """
    Routine to evaluate SSIM metric.

    Parameters
    ----------
    I: 2D Numpy Array, raw image

    I_hat: 2D Numpy Array, denoised image

    mask: 2D Numpy Array, region of interest

    Returns
    -------
	Float number

    """

    I_hat = I_hat*mask
    I_hat = I_hat[I_hat>0]

    I = I[I>0]

    I=I.flatten()
    I_hat=I_hat.flatten()

    L = (np.max(I)-np.min(I))
    c1 = (0.01*L)**2
    c2 = (0.03*L)**2

    return (((2*np.mean(I)*np.mean(I_hat))+c1)*(2*np.cov(I,I_hat)[0][1]+c2)) /  \
	((np.mean(I)**2+np.mean(I_hat)**2+c1)*(np.std(I)**2+np.std(I_hat)**2+c2) )

def SNR(I,I_hat,mask):
    """
    Routine to evaluate SNR metric.

    Parameters
    ----------
    I: 2D Numpy Array, raw image

    I_hat: 2D Numpy Array, denoised image

    mask: 2D Numpy Array, region of interest

    Returns
    -------
	Float number

    """

    I_hat = I_hat*mask
    I_hat = I_hat[I_hat>0]

    I = I[I>0]

    I=I.flatten()
    I_hat=I_hat.flatten()

    J = (I-I_hat)**2
    I_pow = I**2
    return 10*np.log10(np.sum(I_pow)/np.sum(J))

def CoC(I,I_hat,mask):
    """
    Routine to evaluate CoC metric.

    Parameters
    ----------
    I: 2D Numpy Array, raw image

    I_hat: 2D Numpy Array, denoised image

    mask: 2D Numpy Array, region of interest

    Returns
    -------
	Float number

    """

    I = I[I>0]

    I_hat = I_hat*mask
    I_hat = I_hat[I_hat>0]

    I=I.flatten()
    I_hat=I_hat.flatten()

    i = I-np.mean(I)
    i_hat = I_hat-np.mean(I_hat)

    i_pow = i**2
    i_hat_pow = i_hat**2

    return np.sum(i*i_hat)/np.sqrt(np.sum(i_pow)*np.sum(i_hat_pow))

def EPI(I,I_hat,mask):
    """
    Routine to evaluate EPI metric.

    Parameters
    ----------
    I: 2D Numpy Array, raw image

    I_hat: 2D Numpy Array, denoised image

    mask: 2D Numpy Array, region of interest

    Returns
    -------
	Float number

    """

    I_hat = I_hat*mask
    I_hat = I_hat[I_hat>0]

    I = I[I>0]

    I=I.flatten()
    I_hat=I_hat.flatten()

    I = np.gradient(I)
    I_hat = np.gradient(I_hat)

    i = I-np.mean(I)
    i_hat = I_hat-np.mean(I_hat)

    i_pow = i**2
    i_hat_pow = i_hat**2

    return np.sum(i*i_hat)/np.sqrt(np.sum(i_pow)*np.sum(i_hat_pow))
