# nl_means_parallel

A parallel Non-Local Means algorithm  based on scikit-image code.

## Usage

```

>> from nl_means_parallel.filter_parallel import non_local_means_parallel as nlmp
>> data_denoised = nlmp.denoise_nl_means_parallel(data_noisy,os.cpu_count(),1.15,True,5,6)


```

## Installation

You need to build nl_means first, so inside folder nl_means build the codes with cython using the command:

```
python3 setup.py build_ext --inplace

```
## License


The majority of this code is based on the <a href="https://github.com/scikit-image/scikit-image">scikit-image</a> Non-Local Means. Then, as an redistribution with modifications, we acknowledge their <a href= "https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt">LICENSE</a>. Other parts of the code that are not related with the scikit-image can be considered under a MIT License.


## Authors

`nl_means_parallel` was written by `Rafael Henrique `_.
