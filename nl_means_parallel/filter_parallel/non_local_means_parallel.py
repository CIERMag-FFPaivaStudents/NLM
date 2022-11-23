#Rafael Henrique
#Ciermag, IFSC-USP, São Carlos - 12/2022

import numpy as np

from ..nl_means.estimate.estimate_sigma import estimate_sigma
from ..nl_means.filters.non_local_means import denoise_nl_means
from ..nl_means._shared.dtype import img_as_float

from multiprocessing.pool import ThreadPool


def denoise_nl_means_parallel(data_noisy, proc, h_value, fast_mode_value, patch_size_value, patch_distance_value):

	'''
	data_noisy: must be an array with 1,2 or 3 dimension
	
	proc: number of threads
	
	h_value: value of h
	
	fast_mode_value: value of fast_mode
	
	patch_size_value: value of patch size in NLM
	
	patch_distance_value: value of patch distance in NLM

        return: same data_noisy array but after non-local means

	'''


	sigma_est = np.mean(estimate_sigma(data_noisy))

	def performCalc(chunks):
		list = []
		for i in chunks:
			list.append(denoise_nl_means(data_noisy[i,:,:],h=h_value*sigma_est,fast_mode=fast_mode_value,patch_size=patch_size_value,patch_distance=patch_distance_value))
		return list

	data_noisy = img_as_float(data_noisy)

	array_chunks = np.arange(0,data_noisy.shape[0])

	numThreads = proc
	
	pool = ThreadPool(processes = proc)
	
	chunks = np.array_split(array_chunks,proc)
	
	results = pool.map_async(performCalc,chunks)
	
	pool.close()
	
	pool.join()
	
	parallelResults = results.get()
	
	list_=[]
	
	for result in parallelResults:
		for i in result:
			list_.append(i)
			
	return np.asarray(list_)

