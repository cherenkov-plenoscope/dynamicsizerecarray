import dynamicsizerecarray
import pytest
import numpy as np


def test_append_numpy_record_recarray():
    dtype = [("a", "i8"), ("b", "u2"), ("c", "f4")]
    dra = dynamicsizerecarray.DynamicSizeRecarray(dtype=dtype)
    assert dra.shape[0] == 0

    # scalar types
    # ------------

    _tuple = (-12, 14, 3.14)
    dra.append(_tuple)
    assert dra.shape[0] == 1
    assert dra["a"][0] == -12
    assert dra["b"][0] == 14
    assert dra["c"][0] == 3.14

    _dict = {"a": -5, "b": 3, "c": 6.28}
    dra.append(_dict)
    assert dra.shape[0] == 2
    assert dra["a"][1] == -5
    assert dra["b"][1] == 3
    assert dra["c"][1] == 6.28

    _np_void = np.void((15, 2, 1.1), dtype=dtype)
    dra.append(_np_void)
    assert dra.shape[0] == 3
    assert dra["a"][2] == 15
    assert dra["b"][2] == 2
    assert dra["c"][2] == 1.1

    _np_record = np.record((16, 3, 5.5), dtype=dtype)
    dra.append(_np_record)
    assert dra.shape[0] == 4
    assert dra["a"][3] == 16
    assert dra["b"][3] == 3
    assert dra["c"][3] == 5.5

    # array like types
    # ----------------

    # numpy recarray
    ra = np.recarray(shape=2, dtype=dtype)
    ra["a"] = [-17, 18]
    ra["b"] = [3, 7]
    ra["c"] = [2.2, 3.3]
    dra.append(ra)
    assert dra.shape[0] == 6
    for key in dra.dtype.names:
        np.testing.assert_array_equal(dra[key][4:], ra[key])

    # numpy ndarray
    na = np.ndarray(shape=2, dtype=dtype)
    na["a"] = [-33, 21]
    na["b"] = [9, 0]
    na["c"] = [4.0, 2.3]
    dra.append(na)
    assert dra.shape[0] == 8
    for key in dra.dtype.names:
        np.testing.assert_array_equal(dra[key][6:], na[key])

    # list of scalars
    dra.append([_tuple, _dict, _np_void, _np_record])
    assert dra.shape[0] == 12
    np.testing.assert_array_equal(dra[8:12], dra[0:4])


def test_tuple_bad():
    dtype = [("a", "u8")]
    dra = dynamicsizerecarray.DynamicSizeRecarray(dtype=dtype)

    dra.append((1,))

    with pytest.raises(AssertionError) as err:
        dra.append((1, 1))
