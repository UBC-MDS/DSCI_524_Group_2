import numpy as np


def rotate_pyfect(image, n_rot=90):
    """
    This function can be used to apply a rotational transformation on an image.

    The function can be applied on greyscale or 3-channel images. The users can
    choose some degree, theta, which is used in a rotational matrix
    operation to transform the image.

    Parameters
    ----------
    image : numpy.ndarray
        A n*n or n*n*3 numpy array representing an 3-channel image

    n_rot : int
        The number of 90 degree rotations to implement

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

    # Initialize dictionary with each channel being mapped to  it's channel number
    channel_dict = {
        channel: image[channel, :, :] for channel in range(0, image.shape[0])
    }

    # While rotations to do
    while n_rot > 0:
        # For each channel
        for channel in range(0, image.shape[0]):
            # Pull matrix to rotate
            rot_mat = channel_dict[channel]
            # rotate the matrix
            channel_dict[channel] = np.array([list(x)[::-1] for x in zip(*rot_mat)])

        # Update number of rotations left
        n_rot = n_rot - 1

    # Stack rotated matrices to create rotated image
    new_img = np.stack(list(channel_dict.values()))

    return new_img
