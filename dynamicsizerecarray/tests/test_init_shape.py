import dynamicsizerecarray
import pytest
import numpy as np

def test_init_with_shape():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")],
        shape=13,
    )

    assert len(dra) == 13
    assert dra._capacity() == 13


def test_init_negative_shape():
    with pytest.raises(AttributeError):
        dra = dynamicsizerecarray.DynamicSizeRecarray(
            dtype=[("a", "i8")],
            shape=-1,
        )

