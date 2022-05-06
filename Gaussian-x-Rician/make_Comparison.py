#Name: Rafael Henrique
#e-mail: rafaelhenri@usp.br

"""
    Python code for comparing gaussian nlm and rician nlm
"""

#Python libraries
import numpy as np
import sys
import os
from multiprocessing.pool import ThreadPool

sys.path.append('modules')

#modules libraries
import quality_metrics as qm
import denoised_samples as samples
import save_graph
import get_slice



def performCalc(repchunks):

    """
    Routine for evaluate quality metrics in parallel

    Parameters
    ----------
    repchunks: 1D Numpy array
    ---------

    Return:
        A tuple of 4D Numpy array

    """

    CoC = []
    SNR = []
    EPI = []
    SSIM = []
    for chunk in repchunks:
        CoC_tmp = []
        SNR_tmp = []
        EPI_tmp = []
        SSIM_tmp = []
        [(CoC_tmp.append([]), SNR_tmp.append([]),
              EPI_tmp.append([]), SSIM_tmp.append([])) for i in range(8) ]
        for noise in noises:
            DENOISED = samples.generate_denoised_samples_slices(slice, noise+1)
            [(CoC_tmp[i].append(qm.CoC(slice,DENOISED[i],mask)),
              SNR_tmp[i].append(qm.SNR(slice,DENOISED[i],mask)),
              EPI_tmp[i].append(qm.EPI(slice,DENOISED[i],mask)),
              SSIM_tmp[i].append(qm.SSIM(slice,DENOISED[i],mask))) for i in range(8) ]
        CoC.append([CoC_tmp])
        SNR.append([SNR_tmp])
        EPI.append([EPI_tmp])
        SSIM.append([SSIM_tmp])

    return (np.asarray(CoC), np.asarray(SNR), np.asarray(EPI), np.asarray(SSIM))

def separateResults(parallelResults, numThreads):
    """
    Routine for separate each quality metric vector

    Parameters
    ----------
    parallelResults: A tuple of quality of metrics, return of performCalc

    numThreads: 1D Numpy array

    -------
    Return:
        A tuple of 3D Numpy Array
        [iterationsNumber,noises,samples]

    """

    CoC = []
    SNR = []
    EPI = []
    SSIM = []

    for thread in range(numThreads):
        [CoC.append(result[0]) for result in parallelResults[thread][0]]
        [SNR.append(result[0]) for result in parallelResults[thread][1]]
        [EPI.append(result[0]) for result in parallelResults[thread][2]]
        [SSIM.append(result[0]) for result in parallelResults[thread][3]]

    CoC = np.swapaxes(CoC,2,1)
    SNR = np.swapaxes(SNR,2,1)
    EPI = np.swapaxes(EPI,2,1)
    SSIM = np.swapaxes(SSIM,2,1)


    return CoC,SNR, EPI, SSIM

if __name__ == "__main__":

    #get slice and mask
    slice = get_slice.get_axial_Slice_from_Nifti("/../../../../../Volumes_Nifti/otsu_bet/t1_icbm_normal_1mm_pn0_rf0_otsu_brain.nii", 90)
    mask = get_slice.get_axial_Slice_from_Nifti("/../../../../../Volumes_Nifti/otsu_bet/t1_icbm_normal_1mm_pn0_rf0_otsu_brain_mask.nii", 90)

    #amount of noise in %
    noises = np.arange(0,5)


    iterationsNumber = int(sys.argv[1])

    numThreads = os.cpu_count()
    if (iterationsNumber<numThreads):
        numThreads = iterationsNumber
    pool = ThreadPool(processes=numThreads)

    rep=np.arange(iterationsNumber)

    repchunks=np.array_split(rep,numThreads)

    results =pool.map_async(performCalc,repchunks)
    pool.close()
    pool.join()
    parallelResults=results.get()
    parallelResults = np.asarray(parallelResults)


    CoC, SNR, EPI, SSIM = separateResults(parallelResults, numThreads)

    save_graph.samples("results/t1/CoC/","t1_coc_samples","SNR",noises, CoC, iterationsNumber)
    save_graph.samples("results/t1/SNR/","t1_snr_samples","SNR",noises, SNR, iterationsNumber)
    save_graph.samples("results/t1/EPI/","t1_epi_samples","SNR",noises, EPI, iterationsNumber)
    save_graph.samples("results/t1/SSIM/","t1_ssim_samples","SNR",noises, SSIM, iterationsNumber)
