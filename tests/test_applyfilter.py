from picturepyfect import __version__
from picturepyfect import applyfilter as flt
from picturepyfect.applyfilter import (
    FilterTypeException,
    ImageDimensionException,
    FilterDimensionException,
)
from pytest import raises
import numpy as np


def test_version():
    assert __version__ == "0.1.0"


def test_exceptions():
    image = np.arange(1, 26).reshape(5, 5)
    with raises(FilterTypeException):
        flt.filter_pyfect(image, filter_type="invalid_type")

    image = np.zeros((4, 4, 3, 1))
    with raises(ImageDimensionException):
        flt.filter_pyfect(image)

    image = np.zeros((4, 4, 4))
    with raises(ImageDimensionException):
        flt.filter_pyfect(image)

    np.random.seed(2021)
    image = np.arange(1, 26).reshape(5, 5)
    custom_filter = np.random.rand(2, 2, 3, 1)
    with raises(FilterDimensionException):
        flt.filter_pyfect(
            image, filter_type="custom", custom_filter=custom_filter
        )

    np.random.seed(2021)
    custom_filter = np.random.rand(21, 21, 4)
    with raises(FilterDimensionException):
        flt.filter_pyfect(
            image, filter_type="custom", custom_filter=custom_filter
        )

    np.random.seed(2021)
    image = np.random.rand(17, 17, 3)
    np.random.seed(2022)
    custom_filter = np.random.rand(21, 21, 3)
    with raises(ImageDimensionException):
        flt.filter_pyfect(
            image, filter_type="custom", custom_filter=custom_filter
        )

    with raises(FilterTypeException):
        flt.build_filter("Invalid", 7)


# Sub-module filter_pyfect_2D check
def test_filter_pyfect_2D():
    image = np.arange(1, 26).reshape(5, 5)
    kernel = np.ones((2, 2))
    expected_result = np.array(
        [
            [0.0, 0.05555556, 0.11111111, 0.16666667],
            [0.27777778, 0.33333333, 0.38888889, 0.44444444],
            [0.55555556, 0.61111111, 0.66666667, 0.72222222],
            [0.83333333, 0.88888889, 0.94444444, 1.0],
        ]
    )

    assert flt.filter_pyfect_2D(image, kernel).shape == (
        4,
        4,
    ), "Sub-Module filter_pyfect_2D returned unexpected dimension"

    assert (
        np.isclose(expected_result, flt.filter_pyfect_2D(image, kernel))
        == False
    ).sum() == 0, "Sub-Module filter_pyfect_2D returned unexpected value(s)"

    image = np.ones((5, 5))
    kernel = np.ones((2, 2))
    expected_result = np.full((4, 4), 4)

    assert flt.filter_pyfect_2D(image, kernel).shape == (
        4,
        4,
    ), "Sub-Module filter_pyfect_2D returned unexpected dimension"
    assert (
        np.isclose(expected_result, flt.filter_pyfect_2D(image, kernel))
        == False
    ).sum() == 0, "Sub-Module filter_pyfect_2D returned unexpected value(s)"


# Sub-module filter_pyfect_3D check
def test_filter_pyfect_3D():
    image = np.arange(1, 76).reshape(5, 5, 3)
    kernel = np.ones((2, 2, 3))
    expected_result = np.array(
        [
            [0.0, 0.05555556, 0.11111111, 0.16666667],
            [0.27777778, 0.33333333, 0.38888889, 0.44444444],
            [0.55555556, 0.61111111, 0.66666667, 0.72222222],
            [0.83333333, 0.88888889, 0.94444444, 1.0],
        ]
    )

    assert flt.filter_pyfect_3D(image, kernel).shape == (
        4,
        4,
        3,
    ), "Sub-Module filter_pyfect_3D returned unexpected dimension"

    assert (
        np.isclose(
            flt.filter_pyfect_3D(image, kernel)[:, :, 0], expected_result
        )
        == False
    ).sum() == 0, "Sub-Module filter_pyfect_3D 1st channel failed"
    assert (
        np.isclose(
            flt.filter_pyfect_3D(image, kernel)[:, :, 1], expected_result
        )
        == False
    ).sum() == 0, "Sub-Module filter_pyfect_3D 2nd channel failed"
    assert (
        np.isclose(
            flt.filter_pyfect_3D(image, kernel)[:, :, 2], expected_result
        )
        == False
    ).sum() == 0, "Sub-Module filter_pyfect_3D 3rd channel failed"


# Sub-module build_filter check
def test_build_filter():
    expected_blur = np.array(
        [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01], [0.01, 0.01, 0.01]]
    )
    expected_sharpen = np.array(
        [
            [0, 0, 0, -1, 0, 0, 0],
            [0, 0, -1, -1, -1, 0, 0],
            [0, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, 5, -1, -1, -1],
            [0, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, 0, 0],
            [0, 0, 0, -1, 0, 0, 0],
        ]
    )

    assert (
        np.isclose(flt.build_filter("blur", 3), expected_blur) == False
    ).sum() == 0, "Blur filter check failed"
    assert (
        np.isclose(flt.build_filter("sharpen", 7), expected_sharpen) == False
    ).sum() == 0, "Sharpen filter check failed"


# Main-module filter_pyfect check
def test_filter_pyfect():
    image = np.arange(1, 26).reshape(5, 5)
    kernel = np.ones((2, 2))
    expected_result = np.array(
        [
            [0.0, 0.05555556, 0.11111111, 0.16666667],
            [0.27777778, 0.33333333, 0.38888889, 0.44444444],
            [0.55555556, 0.61111111, 0.66666667, 0.72222222],
            [0.83333333, 0.88888889, 0.94444444, 1.0],
        ]
    )

    assert flt.filter_pyfect(
        image, filter_type="custom", custom_filter=kernel
    ).shape == (
        4,
        4,
    ), "Main-Module filter_pyfect returned unexpected dimension"
    assert (
        np.isclose(
            expected_result,
            flt.filter_pyfect(
                image, filter_type="custom", custom_filter=kernel
            ),
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect returned unexpected value(s)"

    image = np.arange(1, 26).reshape(5, 5)
    kernel = np.ones((2, 2, 3))
    expected_result = np.array(
        [
            [0.0, 0.05555556, 0.11111111, 0.16666667],
            [0.27777778, 0.33333333, 0.38888889, 0.44444444],
            [0.55555556, 0.61111111, 0.66666667, 0.72222222],
            [0.83333333, 0.88888889, 0.94444444, 1.0],
        ]
    )
    assert flt.filter_pyfect(
        image, filter_type="custom", custom_filter=kernel
    ).shape == (
        4,
        4,
    ), "Main-Module filter_pyfect returned unexpected dimension"
    assert (
        np.isclose(
            expected_result,
            flt.filter_pyfect(
                image, filter_type="custom", custom_filter=kernel
            ),
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect returned unexpected value(s)"

    image = np.arange(1, 26).reshape(5, 5)
    kernel = np.ones((2, 2))
    expected_result = np.array(
        [
            [0.0, 0.08333333, 0.16666667],
            [0.41666667, 0.5, 0.58333333],
            [0.83333333, 0.91666667, 1.0],
        ]
    )

    assert flt.filter_pyfect(image, filter_type="blur").shape == (
        3,
        3,
    ), "Main-Module filter_pyfect returned unexpected dimension"

    assert (
        np.isclose(
            expected_result, flt.filter_pyfect(image, filter_type="blur")
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect returned unexpected value(s)"

    image = np.arange(1, 76).reshape(5, 5, 3)
    kernel = np.ones((2, 2, 3))
    expected_result = np.array(
        [
            [0.0, 0.05555556, 0.11111111, 0.16666667],
            [0.27777778, 0.33333333, 0.38888889, 0.44444444],
            [0.55555556, 0.61111111, 0.66666667, 0.72222222],
            [0.83333333, 0.88888889, 0.94444444, 1.0],
        ]
    )

    # print(flt.filter_pyfect(image, filter_type="custom", custom_filter=kernel).shape)

    assert flt.filter_pyfect(
        image, filter_type="custom", custom_filter=kernel
    ).shape == (
        4,
        4,
        3,
    ), "Main-Module filter_pyfect returned unexpected dimension"

    assert (
        np.isclose(
            flt.filter_pyfect(
                image, filter_type="custom", custom_filter=kernel
            )[:, :, 0],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 1st channel failed"
    assert (
        np.isclose(
            flt.filter_pyfect(
                image, filter_type="custom", custom_filter=kernel
            )[:, :, 1],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 2nd channel failed"
    assert (
        np.isclose(
            flt.filter_pyfect(
                image, filter_type="custom", custom_filter=kernel
            )[:, :, 2],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 3rd channel failed"

    image = np.arange(1, 76).reshape(5, 5, 3)
    kernel = np.ones((2, 2))
    expected_result = np.array(
        [
            [0.0, 0.05555556, 0.11111111, 0.16666667],
            [0.27777778, 0.33333333, 0.38888889, 0.44444444],
            [0.55555556, 0.61111111, 0.66666667, 0.72222222],
            [0.83333333, 0.88888889, 0.94444444, 1.0],
        ]
    )
    assert flt.filter_pyfect(
        image, filter_type="custom", custom_filter=kernel
    ).shape == (
        4,
        4,
        3,
    ), "Main-Module filter_pyfect returned unexpected dimension"
    assert (
        np.isclose(
            flt.filter_pyfect(
                image, filter_type="custom", custom_filter=kernel
            )[:, :, 0],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 1st channel failed"
    assert (
        np.isclose(
            flt.filter_pyfect(
                image, filter_type="custom", custom_filter=kernel
            )[:, :, 1],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 2nd channel failed"
    assert (
        np.isclose(
            flt.filter_pyfect(
                image, filter_type="custom", custom_filter=kernel
            )[:, :, 2],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 3rd channel failed"

    image = np.arange(1, 76).reshape(5, 5, 3)
    expected_result = np.array(
        [
            [0.0, 0.08333333, 0.16666667],
            [0.41666667, 0.5, 0.58333333],
            [0.83333333, 0.91666667, 1.0],
        ]
    )
    assert flt.filter_pyfect(image, filter_type="blur").shape == (
        3,
        3,
        3,
    ), "Main-Module filter_pyfect returned unexpected dimension"

    assert (
        np.isclose(
            flt.filter_pyfect(image, filter_type="blur")[:, :, 0],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 1st channel failed"
    assert (
        np.isclose(
            flt.filter_pyfect(image, filter_type="blur")[:, :, 1],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 2nd channel failed"
    assert (
        np.isclose(
            flt.filter_pyfect(image, filter_type="blur")[:, :, 2],
            expected_result,
        )
        == False
    ).sum() == 0, "Main-Module filter_pyfect 3rd channel failed"
