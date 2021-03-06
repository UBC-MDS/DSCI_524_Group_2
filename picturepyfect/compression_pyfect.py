import numpy as np

class DimensionError(Exception):
    """ Raised when when a numpy array has the wrong shape. """

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
    
    # Check if the image and kernel_size are valid inputs
    check_values(image, kernel_size)
    
    # Check for a valid pooling_function input
    if pooling_function == "max":
        pool_func = np.max
    elif pooling_function == "min":
        pool_func = np.min
    elif pooling_function == "mean":
        pool_func = np.mean
    else:
        raise ValueError(
            "The pooling_function argument only takes a value of 'max', 'min', or 'mean'."
        )
    
    # If image is not divisible by the kernel_size
    # crop off the right side columns and the bottom rows
    divisible_row = image.shape[0] // kernel_size * kernel_size
    divisible_col = image.shape[1] // kernel_size * kernel_size

    # If image is greyscale, compress just one colour band
    if len(image.shape) == 2:
        image = image[:divisible_row, :divisible_col]
        b1 = pool_band(image, kernel_size, pool_func)
        return b1
    
    # If image is colour, compress all 3 colour bands
    else:
        image = image[:divisible_row, :divisible_col, :]
        
        # Pool the 3 colour bands
        b1 = pool_band(image[:,:,0], kernel_size, pool_func)
        b2 = pool_band(image[:,:,1], kernel_size, pool_func)
        b3 = pool_band(image[:,:,2], kernel_size, pool_func)

        # Combine the 3 colour bands
        return np.dstack((b1,b2,b3))
    
def pool_band(band, kernel_size, pool_func):
    """
    This function is to be used in conjunction with compression_pyfect and compresses
    a single colour band of an image.

    The function applies a lossy pooling algorithm to compress the specified colour band
    and the resulting compressed numpy array is returned.

    Parameters
    ----------
    band : numpy.ndarray
        A n*n numpy array representing a single colour band of an image.

    kernel_size : int
        The size of the kernel to be passed over the image. The resulting filter moving
        across the image will be a 2D array with dimensions kernel_size x kernel_size.

    pooling_function : str
        The pooling algorithm to be used within a kernel. There are three options: "max", "min", and "mean".

    Returns:
    ---------
    numpy.ndarray
        An n*n numpy array representing the compressed image.

    Examples
    --------
    >>> pool_band(image, kernel_size=3, pooling_function="max")
    array([[0.04737957, 0.04648845, 0.04256656, 0.04519495],
       [0.04657273, 0.04489012, 0.04031093, 0.04047667],
       [0.04641026, 0.04106843, 0.04560866, 0.04732271],
       [0.0511907 , 0.04518351, 0.04946411, 0.04030291]])
    """
    
    # Using this forum post as a guide and reference for the below code
    # https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy/42463514#42463514
    
    row = band.shape[0]//kernel_size
    col = band.shape[1]//kernel_size

    # The colour band to pool
    pool_colour_band = band

    # pool along the rows
    pool_colour_band = pool_colour_band.reshape(-1, kernel_size)
    pool_colour_band = pool_func(pool_colour_band, axis=1)
    pool_colour_band = pool_colour_band.reshape(-1, col)

    # rotate and pool along the rows again
    pool_colour_band = np.rot90(pool_colour_band)
    pool_colour_band = pool_colour_band.reshape(-1, kernel_size)
    pool_colour_band = pool_func(pool_colour_band, axis=1)
    pool_colour_band = pool_colour_band.reshape(col, -1)

    # rotate back to proper layout
    pool_colour_band = np.rot90(pool_colour_band, 3)
    return pool_colour_band

def check_values(image, kernel_size):
    """
    This function checks that the image and kernel size are valid inputs
    and raises an error if not.

    Parameters
    ----------
    image : numpy.ndarray
        A n*n or n*n*3 numpy array representing a single channel or 3-channel image.

    kernel_size : int
        The size of the kernel to be passed over the image. The resulting filter moving
        across the image will be a 2D array with dimensions kernel_size x kernel_size.

    Examples
    --------
    >>> check_values(image, kernel_size=3)
    """

    if(not isinstance(image, np.ndarray)):
        raise ValueError(
            "Image must be a numpy array."
        )
    
    if(not isinstance(kernel_size, int)):
        raise ValueError(
            "kernel_size must be a positive integer greater than 0."
        )
        
    if(kernel_size < 1):
        raise ValueError(
            "kernel_size must be a positive integer greater than 0."
        )
    
    # Check if the image is of the correct shape. Greyscale and colour images both accepted
    if (len(image.shape) != 2 and len(image.shape) != 3):
        raise DimensionError(
            "The input image array needs to be of shape n x n, or n x n x 3."
        )
    
    # If the image is of size n x n x n, ensure that the third dimension equals 3.
    if(len(image.shape) == 3):
        if(image.shape[2] != 3):
            raise DimensionError(
                "The input image array needs to be of shape n x n, or n x n x 3."
            )
    
    # Check that the kernel_size is smaller than the image height and width
    if(image.shape[0] < kernel_size or image.shape[1] < kernel_size):
        raise ValueError(
            "The kernel size must not be larger than the height or width of the input image array."
        )