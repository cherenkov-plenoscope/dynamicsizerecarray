import dynamicsizerecarray
import numpy as np


def test_shrink_example():
    dra = dynamicsizerecarray.DynamicSizeRecarray(dtype=[("key", "i8")])
    assert dra.shape[0] == 0
    assert dra._recarray.shape[0] == 2

    for i in range(10):
        dra.append({"key": i})

    assert dra.shape[0] == 10
    assert dra._recarray.shape[0] == 16

    dra.shrink_to_fit()

    assert dra.shape[0] == 10
    assert dra._recarray.shape[0] == 10

    for i in range(10):
        assert dra["key"][i] == i


def test_shrink_shape_zero():
    dra = dynamicsizerecarray.DynamicSizeRecarray(dtype=[("key", "i8")])

    assert dra.shape[0] == 0
    assert dra._recarray.shape[0] == 2

    dra.shrink_to_fit()

    assert dra.shape[0] == 0
    assert dra._recarray.shape[0] == 2  # must keep a minimal capacity
