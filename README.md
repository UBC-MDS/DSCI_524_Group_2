# picturepyfect 

![](https://github.com/UBC-MDS/picturepyfect/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/picturepyfect/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/picturepyfect) [![Deploy](https://github.com/UBC-MDS/picturepyfect/actions/workflows/deploy.yml/badge.svg)](https://github.com/UBC-MDS/picturepyfect/actions/workflows/deploy.yml) [![Documentation Status](https://readthedocs.org/projects/picturepyfect/badge/?version=latest)](https://picturepyfect.readthedocs.io/en/latest/?badge=latest)

A fun Python utility package to make your pictures perfect! The package enables users to process, manipulate, and gather data about their images.

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/ picturepyfect
```

## Features

The package picturepyfect is an image untility package intended to manipulate images through a variety of functions. The intention is that a user with little to no experience can quickly call a function to alter, compress, or print out statistics for an image. Internally, the functions make use of numpy arrays for quick and efficient processes.

There are four main functions planned for development and they are outlined below. Each function can be called on colour images or greyscale images. Additional functions may be added if time permits.

- Function 1 filter_pyfect: With this function, a user can either select a predefined filter or create their own custom filter. The image is then passed through the filter and output for the user to view.

- Function 2 get_property: The goal of this function is to take an image and return statistics related to the different colour bands within the image. These statistics include mean and median values for each channel as well as a plotted histogram of values for each channel.

- Function 3 compression_pyfect: Using a pooling algorithm, this function will apply lossy compression to given image. The user will be able to specify the type of pooling (max, min, or mean) as well as the kernel size.

- Function 4 rotate_pyfect: This final function applies a rotation to a given image and outputs the result. A user can specifiy the number of degrees they wish the image to be rotated.

Image processing is very popular in the Python ecosystem so we are aware that we are not reinventing the wheel with our package, but we hope to gain a deeper understanding of the inner workings of an image package. Specifically, both NumPy and OpenCV have functions that rotate or flip an image (cv2.rotate(), cv2.flip(), np.rot90(), etc). PyTorch has many functions related to convolutional neural networks that both apply pooling layers and filter layers to images which is what we are partially aiming to create with functions 1 and 3. Lastly, the ImageStat module from the Python Imaging Library (PIL) accomplishes many of the tasks outlined in function 2.

## Dependencies

- TODO (important)

## Usage

- TODO

## Documentation

The official documentation is hosted on Read the Docs: https://picturepyfect.readthedocs.io/en/latest/

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/debanandasarkar/picturepyfect/graphs/contributors).

* Chad Neald: @ChadNeald
* Debananda Sarkar: @debanandasarkar
* Dustin Burnham: @dusty736
* Kangbo Lu: @KangboLu

### Workflow

For this project we will be using the GitHub Flow strategy for collaboration. 

### Credits

- This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).

- [Implementing convolutions with stride_tricks](https://jessicastringham.net/2017/12/31/stride-tricks/) by Jessica Stringham
