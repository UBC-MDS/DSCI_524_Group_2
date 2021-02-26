import numpy as np
import matplotlib.pyplot as plt

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

def get_property(image_path, show_formatted_output=True):
    """
    Extract image properties and show visualizations for channel histograms.
    The output properties includes mean and mean of each channel along with the file dimension and total pixels.
    Parameters
    ----------
    image_path : str
        a path to one local image for extract image info and output image property
    show_formatted_output : bool
        a boolean variable to control whether to show the formatted output 
    Returns:
    ---------
    image_property: dictionary
        a dictionary of image properties for dimension of width and height, total pixels,
        and 3 channels' mean and median values separated by channel.
    Examples
    ---------
    >>> get_property("./example_image.jpg", show_formatted_output=True)
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