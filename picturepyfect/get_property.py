import numpy as np
import matplotlib.pyplot as plt

def get_property(image):
    """
    Extract RGB or RGBA image properties. The output properties includes means and medians 
    of RGB channels along with the file dimension and total pixels.
    Parameters
    ----------
    image : numpy.ndarray
        A n*n*3 numpy array to representing 3 or 4-channel RGB or RGBA image
    Returns:
    ---------
    image_property: dictionary
        a dictionary of image properties for dimension of width and height, total pixels,
        and 3 channels' mean and median values separated by channel.
    Examples
    ---------
    >>> get_property(image)
    {dimension: [1280, 720], total_pixels: 921600,
    r_channel: [80, 90], g_channel: [120, 90], b_channel: [155, 160]}
    """
    # error handling for invalid type
    if (type(image) != np.ndarray) or (len(image.shape) != 3) \
        or (image.shape[2] < 3) or (image.shape[2] > 4):
        raise TypeError('Invalid Type: RGB or RGBA image type must be a 3D or 4D numpy array')

    print(image.shape)

    # obtain image properties
    dimension = [image.shape[0], image.shape[1]]
    total_pixels = dimension[0] * dimension[1]
    channel_means = [image[:,:,i].flatten().mean() for i in range(0, 3)]
    channel_medians = [np.median(image[:,:,i].flatten()) for i in range(0, 3)]
    image_properties = {"dimension": dimension, 
                        "total_pixels": total_pixels,
                        "r_channel": [channel_means[0], channel_medians[0]], 
                        "g_channel": [channel_means[1], channel_medians[1]], 
                        "b_channel": [channel_means[2], channel_medians[2]]}

    # return the dictionary of image property
    return image_properties