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

```python
from picturepyfect.compression_pyfect import compression_pyfect
from picturepyfect.rotate_pyfect import rotate_pyfect
from picturepyfect.get_property import get_property
from picturepyfect.applyfilter import filter_pyfect

# Additional imports needed for this example
import os
import numpy as np
from matplotlib.pyplot import imread, imshow
```


```python
# Read in your image
img = imread(os.path.join("beautiful_Vancouver.jpeg"))

# Check your image
imshow(img)
```

    
![png](/img/output_1_1.png)
    

```python
# Rotate the image 90 degrees
rotated_img = rotate_pyfect(img, n_rot=1)
imshow(rotated_img)
```
    
![png](/img/output_2_1.png)
    

```python
# Compress the image
compressed_img = compression_pyfect(img, kernel_size=16, pooling_function="max")
imshow(compressed_img)
```

![png](/img/output_3_1.png)

```python
# Get image properties
get_property(img)
```
    (768, 1024, 3)

    {'dimension': [768, 1024],
     'total_pixels': 786432,
     'r_channel': [116.49133936564128, 104.0],
     'g_channel': [134.16987991333008, 121.0],
     'b_channel': [137.05754470825195, 133.0]}

```python
# Filter the image
filtered_img = filter_pyfect(img, filter_type='blur', filter_size=10)
imshow(filtered_img)
```

![png](/img/output_5_1.png)

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
