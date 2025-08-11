import dynamicsizerecarray
import numpy as np


def make_test_dynrec():
    r = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )
    r.append({"a": 3, "b": 10})
    r.append({"a": 2, "b": 20})
    r.append({"a": 1, "b": 30})

    assert r._recarray.shape[0] > len(
        r
    )  # make sure there is overhead capacity
    return r


def test_setitem_by_str():
    r = make_test_dynrec()

    np.testing.assert_array_equal(r["a"], [3, 2, 1])
    r["a"] = [2, 4, 8]
    np.testing.assert_array_equal(r["a"], [2, 4, 8])

    np.testing.assert_array_equal(r["b"], [10, 20, 30])


def test_setitem_by_int():
    r = make_test_dynrec()
    np.testing.assert_array_equal(r["a"], [3, 2, 1])
    np.testing.assert_array_equal(r["b"], [10, 20, 30])

    # setitem
    r[1] = (4, 17)

    np.testing.assert_array_equal(r["a"], [3, 4, 1])
    np.testing.assert_array_equal(r["b"], [10, 17, 30])
