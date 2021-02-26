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
