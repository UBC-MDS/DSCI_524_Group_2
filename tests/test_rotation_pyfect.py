from picturepyfect.rotation_pyfect import rotation_pyfect
import numpy as np
import pytest

# Generate random images for testing
np.random.seed(42)
test_image = np.random.rand(100, 120, 3)
test_image_1 = rotation_pyfect(test_image, 90)
test_image_2 = rotation_pyfect(test_image, 180)
test_image_3 = rotation_pyfect(test_image, 270)
test_image_4 = rotation_pyfect(test_image, 360)
wrong_dim1 = np.zeros([1, 1])
wrong_dim2 = np.zeros([1, 1, 1])
wrong_dim3 = np.zeros([1, 1, 2])
wrong_dim4 = np.zeros([1, 1, 5])

# test error handling
def test_error():
    with pytest.raises(TypeError):
        rotation_pyfect(None, 90)
    with pytest.raises(TypeError):
        rotation_pyfect(1, 90)
    with pytest.raises(TypeError):
        rotation_pyfect([1], 90)
    with pytest.raises(TypeError):
        rotation_pyfect("string", 90)
    with pytest.raises(TypeError):
        rotation_pyfect(wrong_dim1, 90)
    with pytest.raises(TypeError):
        rotation_pyfect(wrong_dim2, 90)
    with pytest.raises(TypeError):
        rotation_pyfect(wrong_dim3, 90)
    with pytest.raises(TypeError):
        rotation_pyfect(wrong_dim4, 90)
    with pytest.raises(TypeError)
        rotatioin_pyfect(test_image, 1)


# test rotation_pyfect returned dimensions
def test_image_dimensions():
    assert test_image_1.shape == (3, 120, 100)
    assert test_image_2.shape == (3, 100, 120)
    assert test_image_3.shape == (3, 120, 100)
    assert test_image_4.shape == (3, 100, 120)


# test we don't lose any pixel values
def test_image_content():
    assert sum(test_image[0].flatten()) == sum(test_image_1[0].flatten())
    assert sum(test_image[0].flatten()) == sum(test_image_1[0].flatten())
    assert sum(test_image[0].flatten()) == sum(test_image_1[0].flatten())
    assert sum(test_image[1].flatten()) == sum(test_image_1[1].flatten())
    assert sum(test_image[1].flatten()) == sum(test_image_1[1].flatten())
    assert sum(test_image[1].flatten()) == sum(test_image_1[1].flatten())
    assert sum(test_image[2].flatten()) == sum(test_image_1[2].flatten())
    assert sum(test_image[2].flatten()) == sum(test_image_1[2].flatten())
    assert sum(test_image[2].flatten()) == sum(test_image_1[2].flatten())


# test that we don't the rows match the columns after rotation
def test_rows_cols():
    assert (test_image[0, 0, :] == test_image_1[0, :, -1]).all()
    assert (test_image[0, 0, :] == test_image_2[0, -1, :]).all()
    assert (test_image[0, 0, :] == test_image_3[0, :, 1]).all()
    assert (test_image[0, 0, :] == test_image_4[0, :, :]).all()
    assert (test_image[1, 0, :] == test_image_1[1, :, -1]).all()
    assert (test_image[1, 0, :] == test_image_2[1, -1, :]).all()
    assert (test_image[1, 0, :] == test_image_3[1, :, 1]).all()
    assert (test_image[1, 0, :] == test_image_4[1, :, :]).all()
    assert (test_image[2, 0, :] == test_image_1[2, :, -1]).all()
    assert (test_image[2, 0, :] == test_image_2[2, -1, :]).all()
    assert (test_image[2, 0, :] == test_image_3[2, :, 1]).all()
    assert (test_image[2, 0, :] == test_image_4[2, :, :]).all()
