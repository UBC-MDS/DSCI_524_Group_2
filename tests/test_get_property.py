from picturepyfect.get_property import get_property
import matplotlib.pyplot as plt
import pytest

# read test images and get test image properties
pure_red = get_property(plt.imread("./img/red.png"))
pure_green = get_property(plt.imread("./img/green.png"))
pure_blue = get_property(plt.imread("./img/blue.png"))
pure_white = get_property(plt.imread("./img/white.png"))
pure_black = get_property(plt.imread("./img/black.png"))

# test get_property returned dimensions
def test_image_dimensions():
    assert pure_red["dimension"] == [200, 150]
    assert pure_green["dimension"] == [200, 150]
    assert pure_blue["dimension"] == [200, 150]
    assert pure_white["dimension"] == [200, 150]
    assert pure_black["dimension"] == [200, 150]

# test get_property returned total pixels
def test_total_pixels():
    pass

# test get_property returned r channel values
def test_r_channel():
    pass

# test get_property returned g channel values
def test_g_channel():
    pass

# test get_property returned b channel values
def test_b_channel():
    pass