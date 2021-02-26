


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
