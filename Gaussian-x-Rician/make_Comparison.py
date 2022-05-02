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


#get slice and mask
slice = get_slice.get_axial_Slice_from_Nifti("/../../../../../Volumes_Nifti/otsu_bet/t1_icbm_normal_1mm_pn0_rf0_otsu_brain.nii", 90)
mask = get_slice.get_axial_Slice_from_Nifti("/../../../../../Volumes_Nifti/otsu_bet/t1_icbm_normal_1mm_pn0_rf0_otsu_brain_mask.nii", 90)

#CoC parallel
def performCalc_Parallel_CoC(repchunks):
    """
    Routine to perform CoC quality metric in each chunk from multiprocessing.

    Parameters
    ----------
    repchunks: list
        Chunk from iterations.
    Returns
    -------
    vector_chunks: Numpy array
        Numpy array with quality metric from image samples.
    """
    vector_chunks = []
    print(repchunks)
    for k in repchunks:
        vector = []
        for i in range(0,8):
            vector.append([])
        for i in noises:
            DENOISED = samples.generate_denoised_samples_slices(slice, i+1)
            vector[0].append(qm.CoC(slice,DENOISED[0],mask))
            vector[1].append(qm.CoC(slice,DENOISED[1],mask))
            vector[2].append(qm.CoC(slice,DENOISED[2],mask))
            vector[3].append(qm.CoC(slice,DENOISED[3],mask))
            vector[4].append(qm.CoC(slice,DENOISED[4],mask))
            vector[5].append(qm.CoC(slice,DENOISED[5],mask))
            vector[6].append(qm.CoC(slice,DENOISED[6],mask))
            vector[7].append(qm.CoC(slice,DENOISED[7],mask))
        vector_chunks.append(vector)
    return np.asarray(vector_chunks)

#SNR parallel
def performCalc_Parallel_SNR(repchunks):
    """
    Routine to perform SNR quality metric in each chunk from multiprocessing.

    Parameters
    ----------
    repchunks: list
        Chunk from iterations.
    Returns
    -------
    vector_chunks: Numpy array
        Numpy array with quality metric from image samples.
    """
    vector_chunks = []
    print(repchunks)
    for k in repchunks:
        vector = []
        for i in range(0,8):
            vector.append([])
        for i in noises:
            DENOISED = samples.generate_denoised_samples_slices(slice, i+1)
            vector[0].append(qm.SNR(slice,DENOISED[0],mask))
            vector[1].append(qm.SNR(slice,DENOISED[1],mask))
            vector[2].append(qm.SNR(slice,DENOISED[2],mask))
            vector[3].append(qm.SNR(slice,DENOISED[3],mask))
            vector[4].append(qm.SNR(slice,DENOISED[4],mask))
            vector[5].append(qm.SNR(slice,DENOISED[5],mask))
            vector[6].append(qm.SNR(slice,DENOISED[6],mask))
            vector[7].append(qm.SNR(slice,DENOISED[7],mask))
        vector_chunks.append(vector)
    return np.asarray(vector_chunks)

#EPI parallel
def performCalc_Parallel_EPI(repchunks):
    """
    Routine to perform EPI quality metric in each chunk from multiprocessing.

    Parameters
    ----------
    repchunks: list
        Chunk from iterations.
    Returns
    -------
    vector_chunks: Numpy array
        Numpy array with quality metric from image samples.
    """
    vector_chunks = []
    print(repchunks)
    for k in repchunks:
        vector = []
        for i in range(0,8):
            vector.append([])
        for i in noises:
            DENOISED = samples.generate_denoised_samples_slices(slice, i+1)
            vector[0].append(qm.EPI(slice,DENOISED[0],mask))
            vector[1].append(qm.EPI(slice,DENOISED[1],mask))
            vector[2].append(qm.EPI(slice,DENOISED[2],mask))
            vector[3].append(qm.EPI(slice,DENOISED[3],mask))
            vector[4].append(qm.EPI(slice,DENOISED[4],mask))
            vector[5].append(qm.EPI(slice,DENOISED[5],mask))
            vector[6].append(qm.EPI(slice,DENOISED[6],mask))
            vector[7].append(qm.EPI(slice,DENOISED[7],mask))
        vector_chunks.append(vector)
    return np.asarray(vector_chunks)

#SSIM parallel
def performCalc_Parallel_SSIM(repchunks):
    """
    Routine to perform SSIM quality metric in each chunk from multiprocessing.

    Parameters
    ----------
    repchunks: list
        Chunk from iterations.
    Returns
    -------
    vector_chunks: Numpy array
        Numpy array with quality metric from image samples.
    """
    vector_chunks = []
    print(repchunks)
    for k in repchunks:
        vector = []
        for i in range(0,8):
            vector.append([])
        for i in noises:
            DENOISED = samples.generate_denoised_samples_slices(slice, i+1)
            vector[0].append(qm.SSIM(slice,DENOISED[0],mask))
            vector[1].append(qm.SSIM(slice,DENOISED[1],mask))
            vector[2].append(qm.SSIM(slice,DENOISED[2],mask))
            vector[3].append(qm.SSIM(slice,DENOISED[3],mask))
            vector[4].append(qm.SSIM(slice,DENOISED[4],mask))
            vector[5].append(qm.SSIM(slice,DENOISED[5],mask))
            vector[6].append(qm.SSIM(slice,DENOISED[6],mask))
            vector[7].append(qm.SSIM(slice,DENOISED[7],mask))
        vector_chunks.append(vector)
    return np.asarray(vector_chunks)

#Calculate a quality method
def performCalc_old(func):
    """
    Routine to perform parallelResults from func.

    Parameters
    ----------
    func: A performCalc_Parallel_#METRIC function

    Returns
    -------
    vector_chunks: 3D Numpy array
        Numpy array with parallel results processed.

        [iterations,noises,image_samples]
    """
    numThreads = os.cpu_count()
    pool = ThreadPool(processes=numThreads)

    rep=np.arange(iterations)

    repchunks=np.array_split(rep,numThreads)

    results =pool.map_async(func,repchunks)
    pool.close()
    pool.join()
    parallelResults=results.get()
    parallelResults = np.asarray(parallelResults)

    vector = []
    for i in range(0,iterations):
        vector.append([])

    count = 0
    for k in parallelResults: #Não entendi pra que serve essa parte aqui
        if k.size>0:
            for i in noises:
                vector[count].append(k[:,:,i])
            count+=1
    vector = np.asarray(vector)
    return vector[:,:,0,:]

def performCalc(func):

    """


    """

    CoC = []
    SNR = []
    EPI = []
    SSIM = []
    for chunk in repChunk:
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

        return CoC, SNR, EPI, SSIM

if __name__ == "__main__":
   
    #amount of noise in %
    noises = np.arange(0,20)

    iterationsNumber = sys.argv[1]

    numThreads = os.cpu_count()
    pool = ThreadPool(processes=numThreads)

    rep=np.arange(iterations)

    repchunks=np.array_split(rep,numThreads)

    results =pool.map_async(func,repchunks)
    pool.close()
    pool.join()
    parallelResults=results.get()
    parallelResults = np.asarray(parallelResults)

    #perform SNR metric
    vector = performCalc(performCalc_Parallel_SNR)
    save_graph.samples("results/t1/SNR/","t1_snr_samples","SNR",noises, vector, iterations)

    #perform SSIM metric
    vector = performCalc(performCalc_Parallel_SSIM)
    save_graph.samples("results/t1/SSIM/","t1_ssim_samples","SSIM",noises, vector, iterations)

    #perform CoC metric
    vector = performCalc(performCalc_Parallel_CoC)
    save_graph.samples("results/t1/CoC/","t1_coc_samples","CoC",noises, vector, iterations)

    #perform EPI metric
    vector = performCalc(performCalc_Parallel_EPI)
    save_graph.samples("results/t1/EPI/","t1_epi_samples","EPI",noises, vector, iterations)
