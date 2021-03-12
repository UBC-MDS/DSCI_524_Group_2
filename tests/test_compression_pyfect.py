from picturepyfect.compression_pyfect import compression_pyfect
from picturepyfect.compression_pyfect import DimensionError
import pytest
import numpy as np

# Test input arrays
img1 = np.array([[5, 4, 3], [2, 2, 1], [8, 2, 0]])

img2 = np.array(
    [
        [[5, 4, 3], [2, 2, 1], [8, 2, 0]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[0, 1, 0], [0, 1, 0], [1, 10, 0]],
    ]
)

# Expected output arrays
out1 = np.array([[[5, 5, 6]]])
out2 = np.array([[[1, 2, 1]]])
out3 = np.array([[[3, 3.25, 3.25]]])
out4 = np.array([[[8, 10, 9]]])
out5 = np.array([[[0, 1, 0]]])
out6 = np.array([[[3.11, 3.89, 2.44]]])


def test_compression_pyfect():
    # Testing 2D images for all pooling algorithms
    assert (
        compression_pyfect(img1, kernel_size=1, pooling_function="max") == img1
    ).all(), "Max pooling algorithm is failing on 2D image, kernel_size 1."
    assert compression_pyfect(
        img1, kernel_size=2, pooling_function="max"
    ) == np.array(
        [[5]]
    ), "Max pooling algorithm is failing on 2D image, kernel_size 2."
    assert compression_pyfect(
        img1, kernel_size=3, pooling_function="max"
    ) == np.array(
        [[8]]
    ), "Max pooling algorithm is failing on 2D image, kernel_size 3."

    assert (
        compression_pyfect(img1, kernel_size=1, pooling_function="min") == img1
    ).all(), "Min pooling algorithm is failing on 2D image, kernel_size 1."
    assert compression_pyfect(
        img1, kernel_size=2, pooling_function="min"
    ) == np.array(
        [[2]]
    ), "Min pooling algorithm is failing on 2D image, kernel_size 2."
    assert compression_pyfect(
        img1, kernel_size=3, pooling_function="min"
    ) == np.array(
        [[0]]
    ), "Min pooling algorithm is failing on 2D image, kernel_size 3."

    assert (
        compression_pyfect(img1, kernel_size=1, pooling_function="mean")
        == img1
    ).all(), "Mean pooling algorithm is failing on 2D image, kernel_size 1."
    assert compression_pyfect(
        img1, kernel_size=2, pooling_function="mean"
    ) == np.array(
        [[3.25]]
    ), "Mean pooling algorithm is failing on 2D image, kernel_size 2."
    assert compression_pyfect(
        img1, kernel_size=3, pooling_function="mean"
    ) == np.array(
        [[3]]
    ), "Mean pooling algorithm is failing on 2D image, kernel_size 3."

    # Testing 3D images for all pooling algorithms
    assert (
        compression_pyfect(img2, kernel_size=1, pooling_function="max") == img2
    ).all(), "Max pooling algorithm is failing on 3D image, kernel_size 1."
    assert (
        compression_pyfect(img2, kernel_size=2, pooling_function="max") == out1
    ).all(), "Max pooling algorithm is failing on 3D image, kernel_size 2."
    assert (
        compression_pyfect(img2, kernel_size=3, pooling_function="max") == out4
    ).all(), "Max pooling algorithm is failing on 3D image, kernel_size 3."

    assert (
        compression_pyfect(img2, kernel_size=1, pooling_function="min") == img2
    ).all(), "Min pooling algorithm is failing on 3D image, kernel_size 1."
    assert (
        compression_pyfect(img2, kernel_size=2, pooling_function="min") == out2
    ).all(), "Min pooling algorithm is failing on 3D image, kernel_size 2."
    assert (
        compression_pyfect(img2, kernel_size=3, pooling_function="min") == out5
    ).all(), "Min pooling algorithm is failing on 3D image, kernel_size 3."

    assert (
        compression_pyfect(img2, kernel_size=1, pooling_function="mean")
        == img2
    ).all(), "Mean pooling algorithm is failing on 3D image, kernel_size 1."
    assert (
        compression_pyfect(img2, kernel_size=2, pooling_function="mean")
        == out3
    ).all(), "Mean pooling algorithm is failing on 3D image, kernel_size 2."
    assert (
        np.round(
            compression_pyfect(img2, kernel_size=3, pooling_function="mean"), 2
        )
        == out6
    ).all(), "Mean pooling algorithm is failing on 3D image, kernel_size 3."

    # Test pooling_function argument
    with pytest.raises(ValueError):
        compression_pyfect(img1, kernel_size=1, pooling_function="test")

    # Test kernel_size argument
    with pytest.raises(ValueError):
        compression_pyfect(img1, kernel_size=10000, pooling_function="max")
    with pytest.raises(ValueError):
        compression_pyfect(img1, kernel_size=0, pooling_function="max")
    with pytest.raises(ValueError):
        compression_pyfect(img1, kernel_size=-1, pooling_function="max")
    with pytest.raises(ValueError):
        compression_pyfect(img1, kernel_size=1.2, pooling_function="max")
    with pytest.raises(ValueError):
        compression_pyfect(img1, kernel_size="2", pooling_function="max")

    # Test image argument
    with pytest.raises(DimensionError):
        compression_pyfect(
            np.array([2]), kernel_size=1, pooling_function="max"
        )
    with pytest.raises(DimensionError):
        compression_pyfect(
            np.array([[[1]], [[2]], [[3]], [[4]]]),
            kernel_size=1,
            pooling_function="max",
        )
    with pytest.raises(ValueError):
        compression_pyfect(
            image=([[3, 2], [1, 1]]), kernel_size=1, pooling_function="max"
        )
