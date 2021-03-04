import numpy as np


def rotate_pyfect(image, theta=90):
    """
    This function can be used to apply a rotational transformation on an image.

    The function can be applied on greyscale or 3-channel images. The users can
    choose some degree, theta, which is used in a rotational matrix
    operation to transform the image.

    Parameters
    ----------
    image : numpy.ndarray
        A n*n or n*n*3 numpy array representing an 3-channel image

    theta : int
        The degrees to rotate an image.  Default of 90 degrees.

    Returns:
    ---------
    rotated_image: numpy.ndarray
        A n*n or n*n*3 numpy array which is the input image rotated by theta degrees

    Examples
    --------
    >>> np.random.seed(2021)
    >>> image = np.random.rand(2, 2)
    >>> rotate_pyfect(image, deg=90)
    array([[ 0.73336936, -0.60597828],
       [ 0.31267308, -0.13894716]])
    """
    pass
