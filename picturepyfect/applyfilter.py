import numpy as np


class FilterTypeException(Exception):
    """
    Creates custom exception for File Type
    """

    pass


class ImageDimensionException(Exception):
    """
    Creates custom exception for Image Dimension
    """

    pass


class FilterDimensionException(Exception):
    """
    Creates custom exception for Filter Dimension
    """

    pass


def filter_pyfect_2D(image, kernel):
    """
    Performs convolution type filtering using 2D kernel on a 2D numpy array.

    Parameters
    ----------
    image : numpy.ndarray
        A 2D numpy array representing a single channel image

    kernel : numpy.ndarray
        A 2D numpy array representing a convolution filter

    Returns:
    ---------
    filtered_image: numpy.ndarray
        Result of the filtering as a 2D numpy array. Please note that the
        values are scaled so that they are between range of 0 and 1 using minmax scaler
        for plotting stability

    Examples
    --------
    >>> image = np.arange(1, 26).reshape(5, 5)
    >>> kernel = np.ones((2,2))
    >>> filter_pyfect_2D(image, kernel)
    array([[0.        , 0.05555556, 0.11111111, 0.16666667],
           [0.27777778, 0.33333333, 0.38888889, 0.44444444],
           [0.55555556, 0.61111111, 0.66666667, 0.72222222],
           [0.83333333, 0.88888889, 0.94444444, 1.        ]])

    """

    padding_size = 0  # for future enhancement
    stride_size = 1  # for future enhancement

    expanded_input = np.lib.stride_tricks.as_strided(
        image,
        shape=(
            int(
                (image.shape[0] - kernel.shape[0] + (2 * padding_size))
                / stride_size
            )
            + 1,
            int(
                (image.shape[1] - kernel.shape[1] + (2 * padding_size))
                / stride_size
            )
            + 1,
            kernel.shape[0],
            kernel.shape[1],
        ),
        strides=(
            image.strides[0],
            image.strides[1],
            image.strides[0],
            image.strides[1],
        ),
        writeable=False,
    )
    filtered_image = (expanded_input * kernel).sum(axis=(2, 3))

    if filtered_image.max() != filtered_image.min():
        filtered_image = (filtered_image - filtered_image.min()) / (
            filtered_image.max() - filtered_image.min()
        )
    return filtered_image


def filter_pyfect_3D(image, kernel):
    """
    Performs convolution type filtering using 3D kernel on a 3D numpy array.

    Both the kernel and image should have 3 channels in the 3rd dimension.

    Parameters
    ----------
    image : numpy.ndarray
        A 3D numpy array representing a 3 channel image

    kernel : numpy.ndarray
        A 3D numpy array representing a 3D convolution filter with 3 channels

    Returns:
    ---------
    filtered_image: numpy.ndarray
        Result of the filtering as a 3D numpy array. Please note that the
        values are scaled so that they are between range of 0 and 1 using minmax scaler
        within each channel for plotting stability

    Examples
    --------
    >>> image = np.arange(1, 76).reshape(5, 5, 3)
    >>> kernel = np.ones((2,2,3))
    >>> result = filter_pyfect_3D(image, kernel)
    >>> result[:,:,0]
    array([[0.        , 0.05555556, 0.11111111, 0.16666667],
       [0.27777778, 0.33333333, 0.38888889, 0.44444444],
       [0.55555556, 0.61111111, 0.66666667, 0.72222222],
       [0.83333333, 0.88888889, 0.94444444, 1.        ]])

    >>> result[:,:,1]
    array([[0.        , 0.05555556, 0.11111111, 0.16666667],
       [0.27777778, 0.33333333, 0.38888889, 0.44444444],
       [0.55555556, 0.61111111, 0.66666667, 0.72222222],
       [0.83333333, 0.88888889, 0.94444444, 1.        ]])

    >>> result[:,:,2]
    array([[0.        , 0.05555556, 0.11111111, 0.16666667],
       [0.27777778, 0.33333333, 0.38888889, 0.44444444],
       [0.55555556, 0.61111111, 0.66666667, 0.72222222],
       [0.83333333, 0.88888889, 0.94444444, 1.        ]])

    """

    padding_size = 0  # for future enhancement
    stride_size = 1  # for future enhancement

    filtered_image = np.zeros(
        (
            int(
                (image.shape[0] - kernel.shape[0] + (2 * padding_size))
                / stride_size
            )
            + 1,
            int(
                (image.shape[1] - kernel.shape[1] + (2 * padding_size))
                / stride_size
            )
            + 1,
            3,
        )
    )

    for k in range(3):
        temp_image = image[:, :, k]
        temp_kernel = kernel[:, :, k]
        filtered_image[:, :, k] = filter_pyfect_2D(temp_image, temp_kernel)

    return filtered_image


def build_filter(kernel_type, kernel_size):
    """
    This function can be used to build predefined filters.

    Parameters
    ----------
    filter_type : string
        One of the following values:
            blur: Used to blur the picture
            sharpen: Used to increase the sharpness of the image
        More options will be added as enhancements

    filter_size : int
        An integer determining the filter size.

    Returns:
    ---------
    image_property: numpy.ndarray
        A kernel_size * kernel_size numpy array representing the filter.

    Examples
    --------
    >>> build_filter("blur", 3)
    array([[0.01, 0.01, 0.01],
       [0.01, 0.01, 0.01],
       [0.01, 0.01, 0.01]])

    >>> build_filter("sharpen", 7)
    array([[ 0,  0,  0, -1,  0,  0,  0],
       [ 0,  0, -1, -1, -1,  0,  0],
       [ 0, -1, -1, -1, -1, -1,  0],
       [-1, -1, -1,  5, -1, -1, -1],
       [ 0, -1, -1, -1, -1, -1,  0],
       [ 0,  0, -1, -1, -1,  0,  0],
       [ 0,  0,  0, -1,  0,  0,  0]])

    """
    if kernel_type == "blur":
        kernel = np.full((kernel_size, kernel_size), 0.01)

    elif kernel_type == "sharpen":
        kernel = np.full((kernel_size, kernel_size), -1)
        kernel[int(kernel_size / 2), int(kernel_size / 2)] = 5
        for i in range(int(kernel_size / 2)):
            kernel[i, np.arange(int(kernel_size / 2) - i)] = 0
            kernel[
                i, np.arange(int(kernel_size / 2) + 1 + i, int(kernel_size))
            ] = 0
            kernel[
                int(kernel_size) - i - 1, np.arange(int(kernel_size / 2) - i)
            ] = 0
            kernel[
                int(kernel_size) - i - 1,
                np.arange(int(kernel_size / 2) + 1 + i, int(kernel_size)),
            ] = 0

    else:
        raise FilterTypeException(f"Invalid filter_type.")

    return kernel


def filter_pyfect(
    image, filter_type="blur", filter_size=3, custom_filter=None
):
    """
    This function can be used to apply predefined or custom filters on an image.

    The function can be applied on single channel or 3-channel images. The users can
    choose from predefined filters or can create their new filters. This can be used
    for various purposes like entertainment application or visualization of
    convolutional neural network.

    Parameters
    ----------
    image : numpy.ndarray
        A n1*n2 or n1*n2*3 numpy array to representing single channel or 3-channel image

    filter_type : string
        One of the following values:
            blur: Used to blur the picture
            sharpen: Used to increase the sharpness of the image
            custom: Allows users to use their own filter
        More options will be added as enhancements

    filter_size : int
        An integer determining the filter size.
        This is used if the filter_type is not custom. Default: 3

    custom_filter: numpy.ndarray
        A k1*k2 or k1*k2*3 numpy array allows users to pass their own filter. This is only
        used if the users select filter_type = "custom"

    Returns:
    ---------
    filtered_image: numpy.ndarray
        A numpy array representing the transformed image.

    Examples
    --------
    >>> image = np.arange(1, 26).reshape(5, 5)
    >>> kernel = np.ones((2,2))
    >>> filter_pyfect(image, filter_type="custom", custom_filter=kernel)
    array([[0.        , 0.05555556, 0.11111111, 0.16666667],
       [0.27777778, 0.33333333, 0.38888889, 0.44444444],
       [0.55555556, 0.61111111, 0.66666667, 0.72222222],
       [0.83333333, 0.88888889, 0.94444444, 1.        ]])

    >>> image = np.arange(1, 76).reshape(5, 5, 3)
    >>> kernel = np.ones((2,2,3))
    >>> filter_pyfect(image, filter_type="custom", custom_filter=kernel)[:,:,1]
    array([[0.        , 0.05555556, 0.11111111, 0.16666667],
       [0.27777778, 0.33333333, 0.38888889, 0.44444444],
       [0.55555556, 0.61111111, 0.66666667, 0.72222222],
       [0.83333333, 0.88888889, 0.94444444, 1.        ]])

    >>> image = np.arange(1, 26).reshape(5, 5)
    >>> filter_pyfect(image, filter_type="blur")
    array([[0.        , 0.08333333, 0.16666667],
       [0.41666667, 0.5       , 0.58333333],
       [0.83333333, 0.91666667, 1.        ]])

    """
    valid_filters = ["blur", "sharpen", "custom"]
    valid_dimensions = [2, 3]

    if not (filter_type in valid_filters):
        raise FilterTypeException(
            f"Invalid filter_type. Please use one out of {valid_filters}"
        )

    if (not (image.ndim in valid_dimensions)) or (
        image.ndim == 3 and image.shape[2] != 3
    ):
        raise ImageDimensionException(
            f"Invalid dimension of the image. Please use 2D or 3D images. In case of 3D images, there should be 3 channels"
        )

    if filter_type == "custom" and (
        (not (custom_filter.ndim in valid_dimensions))
        or (custom_filter.ndim == 3 and custom_filter.shape[2] != 3)
    ):
        raise FilterDimensionException(
            f"Invalid dimension of the filter. Please use 2D or 3D filters. In case of 3D filters, there should be 3 channels"
        )

    if (
        filter_type == "custom"
        and (
            image.shape[0] <= custom_filter.shape[0]
            or image.shape[1] <= custom_filter.shape[1]
        )
    ) or (
        filter_type != "custom"
        and (image.shape[0] <= filter_size or image.shape[1] <= filter_size)
    ):
        raise ImageDimensionException(
            "Image size has to be bigger than filter size"
        )

    if image.ndim == 2:
        if filter_type != "custom":
            kernel = build_filter(filter_type, filter_size)
        else:
            if custom_filter.ndim == 2:
                kernel = custom_filter.copy()
            else:
                kernel = custom_filter[:, :, 0]
        filtered_image = filter_pyfect_2D(image, kernel)
    else:
        if filter_type != "custom":
            temp = build_filter(filter_type, filter_size)
            kernel = temp[:, :, np.newaxis] + np.zeros(
                (temp.shape[0], temp.shape[1], 3)
            )

        else:
            if custom_filter.ndim == 3:
                kernel = custom_filter.copy()
            else:
                kernel = custom_filter[:, :, np.newaxis] + np.zeros(
                    (custom_filter.shape[0], custom_filter.shape[1], 3)
                )

        filtered_image = filter_pyfect_3D(image, kernel)

    return filtered_image
