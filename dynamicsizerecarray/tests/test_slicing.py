import dynamicsizerecarray
import pytest


def test_slice():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )

    for i in range(100):
        dra.append_record({"a": 2 * i, "b": i})

    with pytest.raises(IndexError):
        _ = dra[200]

    ret = dra[200:300]
    assert ret.shape[0] == 0

    s = dra[10:20]
    for idx, val in enumerate(range(10, 20)):
        assert s["b"][idx] == val

    s = dra[10:2:20]
    for idx, val in enumerate(range(10, 2, 20)):
        assert s["b"][idx] == val
