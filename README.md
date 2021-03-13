# picturepyfect 

![](https://github.com/UBC-MDS/picturepyfect/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/picturepyfect/branch/master/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/picturepyfect) [![Deploy](https://github.com/UBC-MDS/picturepyfect/actions/workflows/deploy.yml/badge.svg)](https://github.com/UBC-MDS/picturepyfect/actions/workflows/deploy.yml) [![Documentation Status](https://readthedocs.org/projects/picturepyfect/badge/?version=latest)](https://picturepyfect.readthedocs.io/en/latest/?badge=latest)

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

[tool.poetry.dependencies]  
* python = "^3.8"  
* numpy = "^1.20.1"  
* matplotlib = "^3.3.4"  

[tool.poetry.dev-dependencies]  
* pytest = "^6.2.2"  
* pytest-cov = "^2.11.1"  
* codecov = "^2.1.11"  
* python-semantic-release = "^7.15.0"  
* flake8 = "^3.8.4"  
* Sphinx = "^3.5.2"  
* sphinxcontrib-napoleon = "^0.7"  

## Usage

* `filter_pyfect_2D(image, kernel)`
```
    Examples
    --------
    >>> image = np.arange(1, 26).reshape(5, 5)
    >>> kernel = np.ones((2,2))
    >>> filter_pyfect_2D(image, kernel)
    array([[0.        , 0.05555556, 0.11111111, 0.16666667],
           [0.27777778, 0.33333333, 0.38888889, 0.44444444],
           [0.55555556, 0.61111111, 0.66666667, 0.72222222],
           [0.83333333, 0.88888889, 0.94444444, 1.        ]])
```

* `rotate_pyfect(image, n_rot=1)`
```
    Examples
    --------
    >>> np.random.seed(42)
    >>> image = np.random.rand(2, 2, 1)
    >>> rotate_pyfect(image, deg=1)
    array([[[0.73199394],
        [0.37454012]],
       [[0.59865848],
        [0.95071431]]])
```

* `compression_pyfect(image, kernel_size=2, pooling_function="max")`
```
    Examples
    --------
    >>> compression_pyfect(image, kernel_size=3, pooling_function="max")
    array([[0.04737957, 0.04648845, 0.04256656, 0.04519495],
       [0.04657273, 0.04489012, 0.04031093, 0.04047667],
       [0.04641026, 0.04106843, 0.04560866, 0.04732271],
       [0.0511907 , 0.04518351, 0.04946411, 0.04030291]])
```

* `get_property(image)`
```
    Examples
    ---------
    >>> get_property(image)
    {dimension: [1280, 720], total_pixels: 921600,
    r_channel: [80, 90], g_channel: [120, 90], b_channel: [155, 160]}
```


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
