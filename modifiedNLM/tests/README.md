# tests

A test for modified Non-Local Means algorithm for Rician noise based on scikit-image code.

## Usage

```
Usage example: "example.py" file:

  >>> from modifiedNLM import tests
  >>> tests.test_pass()

  "example.py" is in the same folder as modifiedNLM.

  parent_folder
              |__example.py
              |__modifiedNLM

              If code above generates an error there is something wrong with this package.

              ----------------------------------------------

              About test sample images:

              input_image_test is a nifti image, with an axial slice of a brain in T1 weighted.

              expected_image_test is a nifti image, input_image_test after denoising with rician_denoised_nl_means.

              expected_image_test is the value expected after denoising.

```

## Authors

`tests` was written by `Rafael Henrique <rafaelhenri@usp.br>`_.
