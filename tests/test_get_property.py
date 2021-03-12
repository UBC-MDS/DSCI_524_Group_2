from picturepyfect.get_property import get_property
import matplotlib.pyplot as plt
import numpy as np
import pytest

# read test images and get test image properties
pure_red = get_property(plt.imread("./img/red.png"))
pure_green = get_property(plt.imread("./img/green.png"))
pure_blue = get_property(plt.imread("./img/blue.png"))
pure_white = get_property(plt.imread("./img/white.png"))
pure_black = get_property(plt.imread("./img/black.png"))
pure_black = get_property(plt.imread("./img/black.png"))
wrong_dim1 = np.zeros([1, 1])
wrong_dim2 = np.zeros([1, 1, 1])
wrong_dim3 = np.zeros([1, 1, 2])
wrong_dim4 = np.zeros([1, 1, 5])

# test error handling
def test_error():
    with pytest.raises(TypeError):
        get_property(None)
    with pytest.raises(TypeError):
        get_property(1)
    with pytest.raises(TypeError):
        get_property([1])
    with pytest.raises(TypeError):
        get_property("string")
    with pytest.raises(TypeError):
        get_property(wrong_dim1)
    with pytest.raises(TypeError):
        get_property(wrong_dim2)
    with pytest.raises(TypeError):
        get_property(wrong_dim3)
    with pytest.raises(TypeError):
        get_property(wrong_dim4)


# test get_property returned dimensions
def test_image_dimensions():
    assert pure_red["dimension"] == [200, 150]
    assert pure_green["dimension"] == [200, 150]
    assert pure_blue["dimension"] == [200, 150]
    assert pure_white["dimension"] == [200, 150]
    assert pure_black["dimension"] == [200, 150]


# test get_property returned total pixels
def test_total_pixels():
    assert pure_red["total_pixels"] == 30000
    assert pure_green["total_pixels"] == 30000
    assert pure_blue["total_pixels"] == 30000
    assert pure_white["total_pixels"] == 30000
    assert pure_black["total_pixels"] == 30000


# test get_property returned r channel values
def test_r_channel():
    assert pure_red["r_channel"] == [1.0, 1.0]
    assert pure_green["r_channel"] == [0.0, 0.0]
    assert pure_blue["r_channel"] == [0.0, 0.0]
    assert pure_white["r_channel"] == [1.0, 1.0]
    assert pure_black["r_channel"] == [0.0, 0.0]


# test get_property returned g channel values
def test_g_channel():
    assert pure_red["g_channel"] == [0.0, 0.0]
    assert pure_green["g_channel"] == [1.0, 1.0]
    assert pure_blue["g_channel"] == [0.0, 0.0]
    assert pure_white["g_channel"] == [1.0, 1.0]
    assert pure_black["g_channel"] == [0.0, 0.0]


# test get_property returned b channel values
def test_b_channel():
    assert pure_red["b_channel"] == [0.0, 0.0]
    assert pure_green["b_channel"] == [0.0, 0.0]
    assert pure_blue["b_channel"] == [1.0, 1.0]
    assert pure_white["b_channel"] == [1.0, 1.0]
    assert pure_black["b_channel"] == [0.0, 0.0]