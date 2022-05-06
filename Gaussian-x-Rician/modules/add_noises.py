#Name: Rafael Henrique
#e-mail: rafaelhenri@usp.br

"""
    Python code for adding gaussian or rician noise
"""

import numpy as np

def gaussian_noise(vector,error_percent):
    """
    Routine to perform adding gaussian noise.

    Parameters
    ----------
    vector: 2D Numpy Array

    error_percent: Integer number between 1 and 100

    Returns
    -------
    Numpy array
        2D Numpy array with noise.
    """
    sigma = error_percent*np.mean(vector)
    sigma = sigma/100

    noise = np.random.normal(0,sigma,vector.shape)

    return vector+noise

def rician_noise(vector,error_percent):
    """
    Routine to perform adding rician noise.

    Parameters
    ----------
    vector: 2D Numpy Array

    error_percent: Integer number between 1 and 100

    Returns
    -------
    Numpy array
        2D Numpy array with noise.
    """

    sigma = error_percent*np.mean(vector)
    sigma = sigma/100

    noise1 = np.random.normal(0,sigma,vector.shape)
    noise2 = np.random.normal(0,sigma,vector.shape)

    return np.sqrt(((vector/np.sqrt(2))+noise1)**2+((vector/np.sqrt(2))+noise2)**2)
