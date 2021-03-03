import numpy as np
import matplotlib.pyplot as plt

def get_property(image):
    """
    Extract image properties. The output properties includes mean and mean of each channel 
    along with the file dimension and total pixels.
    Parameters
    ----------
    image : numpy.ndarray
        A n*n or n*n*3 numpy array to representing single channel or 3-channel image
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