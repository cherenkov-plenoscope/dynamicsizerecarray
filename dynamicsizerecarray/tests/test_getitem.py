import dynamicsizerecarray
import numpy as np


def make_test_dynrec():
    r = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )
    r.append({"a": 123, "b": 12})
    r.append({"a": -23, "b": 1})
    return r


def test_getitem_by_str():
    r = make_test_dynrec()

    a_column = r["a"]
    assert a_column.dtype == "i8"
    assert len(a_column) == 2
    assert a_column[0] == 123
    assert a_column[1] == -23

    b_column = r["b"]
    assert b_column.dtype == "u2"
    assert len(b_column) == 2
    assert b_column[0] == 12
    assert b_column[1] == 1


def test_getitem_by_int():
    r = make_test_dynrec()

    row_0 = r[0]
    assert row_0.dtype == [("a", "i8"), ("b", "u2")]
    assert row_0["a"] == 123
    assert row_0["b"] == 12

    row_1 = r[1]
    assert row_1.dtype == [("a", "i8"), ("b", "u2")]
    assert row_1["a"] == -23
    assert row_1["b"] == 1
