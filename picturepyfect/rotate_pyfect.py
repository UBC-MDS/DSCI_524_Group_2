import numpy as np


def rotate_pyfect(image, n_rot=1):
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
        A n*n or n*n*3 numpy array which is the input image rotated by a multiple of 90 degrees

    Examples
    --------
    >>> np.random.seed(42)
    >>> image = np.random.rand(2, 2, 1)
    >>> rotate_pyfect(image, deg=1)
    array([[[0.73199394],
        [0.37454012]],

       [[0.59865848],
        [0.95071431]]])
    """

    # error handling for invalid type
    if type(image) != np.ndarray:
        raise TypeError("Invalid Type: Image must be a numpy array")

    # error handling for invalid n_rot
    if type(n_rot) != int:
        raise TypeError("Invalid Type: n_rot must be integer")

    # error handling for invalid n_rot
    if n_rot not in set([1, 2, 3, 4]):
        raise TypeError("Invalid Type: n_rot must be between 1 and 4")

    # error handling for invalid dimensions
    if len(image.shape) != 3:
        raise TypeError("Invalid Type: Image must by 3 dimensional")

    # Initialize dictionary with each channel being mapped to  it's channel number
    channel_dict = {
        channel: image[:, :, channel] for channel in range(0, image.shape[2])
    }

    # While rotations to do
    while n_rot > 0:
        # For each channel
        for channel in range(0, image.shape[2]):
            # Pull matrix to rotate
            rot_mat = channel_dict[channel]
            # rotate the matrix
            channel_dict[channel] = np.array([row[::-1] for row in zip(*rot_mat)])

        # Update number of rotations left
        n_rot = n_rot - 1

    # Stack rotated matrices to create rotated image
    new_img = np.dstack(list(channel_dict.values()))

    return new_img
