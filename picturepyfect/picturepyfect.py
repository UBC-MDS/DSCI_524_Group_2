import numpy as np
import matplotlib.pyplot as plt
import os


def filter_pyfect(image, filter_type="blur", filter_size=3, custom_filter=None):
    """
    This function can be used to apply predefined or custom filters on an image.

    The function can be applied on single channel or 3-channel images. The users can
    choose from predefined filters or can create their new filters. This can be used
    for various purposes like entertainment application or visualization of
    convolutional neural network.

    Parameters
    ----------
    image : numpy.ndarray
        A n*n or n*n*3 numpy array to representing single channel or 3-channel image

    filter_type : string
        One of the following values:
            blur: Used to blur the picture
            grayscale: Used to create a single channel image
            crop-sides: Used to crop pixels from all sides
            custom: Allows users to use their own filter
        More options will be added as enhancements

    filter_size : int
        An integer determining the filter size. Default: 3

    custom_filter : numpy.ndarray
        A k*k or k*k*3 numpy array allows users to pass their own filter. This is only
        used if the users select filter_type = "custom"
    
    Returns:
    ---------
    image_property: numpy.ndarray
        A numpy array representing the transformed image.

    Examples
    --------
    >>> np.random.seed(2021)
    >>> image = np.random.rand(6, 6)
    >>> filter_pyfect(image, filter_type="blur", filter_size=3, custom_filter=None)
    array([[0.04737957, 0.04648845, 0.04256656, 0.04519495],
       [0.04657273, 0.04489012, 0.04031093, 0.04047667],
       [0.04641026, 0.04106843, 0.04560866, 0.04732271],
       [0.0511907 , 0.04518351, 0.04946411, 0.04030291]])

    """
    pass


def get_property(image, show_formatted_output=True):
    """
    Extract image properties and show visualizations for channel histograms.
    The output properties includes mean and mean of each channel along with the file dimension and total pixels.
    Parameters
    ----------
    image : numpy.ndarray
        A n*n or n*n*3 numpy array to representing single channel or 3-channel image
    show_formatted_output : bool
        a boolean variable to control whether to show the formatted output
    Returns:
    ---------
    image_property: dictionary
        a dictionary of image properties for dimension of width and height, total pixels,
        and 3 channels' mean and median values separated by channel.
    Examples
    ---------
    >>> get_property(image, show_formatted_output=True)
    {dimension: [1280, 720], total_pixels: 921600,
    r_channel: [80, 90], g_channel: [120, 90], b_channel: [155, 160]}

    ===============================
    SHOW FORMATTED IMAGE PROPERTIES
    ===============================
    Image dimension: 1280 x 720
    Total pixels: 921600 pixels
    R Channel:
        Mean: 80
        Median: 90
    G Channel:
        Mean: 120
        Median: 90
    B Channel:
        Mean: 155
        Median: 160

    Show Histograms for Each Channel:
    """
    pass

def compression_pyfect(image, kernel_size=2, pooling_function="max"):
    """
    This function uses a lossy pooling algorithm to compress an image.
    
    The function can be applied to single channel or 3-channel images. The user passes
    an image which is to be compressed and the resulting compressed numpy array is returned.
    The user can also specify the pooling algorithm to be used and the size of the kernel
    to apply over the image.

    Parameters
    ----------
    image : numpy.ndarray
        A n*n or n*n*3 numpy array representing a single channel or 3-channel image.

    kernel_size : int
        The size of the kernel to be passed over the image. The resulting filter moving
        across the image will be a 2D array with dimensions kernel_size x kernel_size.
        Default: 2
               
    pooling_function : str
        The pooling algorithm to be used within a kernel. There are three options: "max", "min", and "mean".
        Default: "max"

    Returns:
    ---------
    numpy.ndarray
        A numpy array representing the compressed image.

    Examples
    --------
    >>> compression_pyfect(image, kernel_size=3, pooling_function="max")
    array([[0.04737957, 0.04648845, 0.04256656, 0.04519495],
       [0.04657273, 0.04489012, 0.04031093, 0.04047667],
       [0.04641026, 0.04106843, 0.04560866, 0.04732271],
       [0.0511907 , 0.04518351, 0.04946411, 0.04030291]])       
    """
    pass

def rotate_pyfect(image, theta=90):
    """
    This function can be used to apply a rotational transformation on an image.

    The function can be applied on m-channel images. The users can
    choose some degree, theta, which is used in a rotational matrix
    operation to transform the image.

    Parameters
    ----------
    image : numpy.ndarray
        A n*n or n*n*m numpy array representing an m-channel image

    theta : int
        The degrees to rotate an image.  Default of 90 degrees.

    Returns:
    ---------
    rotated_image: numpy.ndarray
        A n*n or n*n*m numpy array which is the input image rotated by theta degrees

    Examples
    --------
    >>> np.random.seed(2021)
    >>> image = np.random.rand(2, 2)
    >>> rotate_pyfect(image, deg=90)
    array([[ 0.73336936, -0.60597828],
       [ 0.31267308, -0.13894716]])
    """
    pass
