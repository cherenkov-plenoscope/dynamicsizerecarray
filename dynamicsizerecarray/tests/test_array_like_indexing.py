import dynamicsizerecarray
import pytest
import numpy as np


def make_example_dynamicsizerecarray():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )
    for i in range(100):
        dra.append({"a": 2 * i, "b": i})
    return dra


def test_list_of_three():
    dra = make_example_dynamicsizerecarray()
    ret = dra[[1, 2, 3]]
    assert ret.shape[0] == 3
    np.testing.assert_array_equal(ret["a"], [2, 4, 6])
    np.testing.assert_array_equal(ret["b"], [1, 2, 3])


def test_idx_out_of_range():
    dra = make_example_dynamicsizerecarray()
    with pytest.raises(IndexError):
        _ = dra[[1000, 2000, 3000]]
