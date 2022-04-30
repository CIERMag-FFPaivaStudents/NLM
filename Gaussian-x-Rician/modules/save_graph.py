#Name: Rafael Henrique
#e-mail: rafaelhenri@usp.br

"""
    Python code for saving graph and numpy array
"""

import numpy as np
import matplotlib.pyplot as plt


def samples(path_to_save,save_name,method_name,noises, METHOD, it):
    """
    Routine for ploting graph and saving numpy array for a quality metric.

    Parameters
    ----------
    path_to_save:
                    String, path for saving graph
    save_name:
                    String, name for your png image
    method_name:
                    Quality Metric name
    noises:
                    1D Numpy Array
    METHOD:
                    3D Numpy Array, with [iterations,noises,image_samples]
    it:
                    Integer, number of iterations
    Returns
    -------

    """

    #save quality metric numpy array
    np.save(path_to_save+save_name+'.npy', METHOD)



    error_bar = []
    for i in range(0,8):
        error_bar.append([])

    VEC = []
    for i in range(0,8):
        VEC.append([])

    for l in range(0,8):
        for i in noises:
            it_vector = []
            for k in range(0,it):
                it_vector.append(METHOD[k,i,l])
            it_vector = np.asarray(it_vector)
            VEC[l].append(np.add.reduce(it_vector)/it)
            error_bar[l].append(np.std(VEC[l]))


    noises=noises+1

    plt.figure(figsize=(10,10))

    plt.errorbar(noises,VEC[0],yerr=np.asarray(error_bar[0]),ecolor="black",ls="dotted", color="red", label="RrR")
    plt.errorbar(noises,VEC[1], yerr=np.asarray(error_bar[1]),ecolor='black',ls='dotted', color='blue', label="RrG")
    plt.errorbar(noises,VEC[2], yerr=np.asarray(error_bar[2]),ecolor='black',ls='dotted', color='green', label="RgG")
    plt.errorbar(noises,VEC[3], yerr=np.asarray(error_bar[3]),ecolor='black',ls='dotted', color='black', label="RgR")
    plt.errorbar(noises,VEC[4], yerr=np.asarray(error_bar[4]),ecolor='black', color='red', label="GrR")
    plt.errorbar(noises,VEC[5], yerr=np.asarray(error_bar[5]),ecolor='black', color='blue', label="GrG")
    plt.errorbar(noises,VEC[6], yerr=np.asarray(error_bar[6]),ecolor='black', color='green', label="GgG")
    plt.errorbar(noises,VEC[7], yerr= np.asarray(error_bar[7]),ecolor='black', color='black', label="GgR")

    plt.legend()


    plt.xlabel("noises (%)");
    plt.ylabel(method_name);
    plt.title(method_name);
    plt.xticks(noises)
    plt.xlim([1,np.max(noises)])

    plt.savefig(path_to_save+save_name+".png")
    plt.cla()
    plt.clf()
