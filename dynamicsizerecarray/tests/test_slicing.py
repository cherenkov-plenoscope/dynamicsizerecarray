import dynamicsizerecarray
import pytest


def make_example_dynamicsizerecarray():
    dra = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("a", "i8"), ("b", "u2")]
    )
    for i in range(100):
        dra.append({"a": 2 * i, "b": i})
    return dra


def test_slice_start():
    dra = make_example_dynamicsizerecarray()
    ret = dra[50:]
    assert ret.shape[0] == 50


def test_slice_stop():
    dra = make_example_dynamicsizerecarray()
    ret = dra[:50]
    assert ret.shape[0] == 50


def test_slice_start_and_stop():
    dra = make_example_dynamicsizerecarray()
    ret = dra[25:50]
    assert ret.shape[0] == 25


def test_slice_start_and_stop_and_step():
    dra = make_example_dynamicsizerecarray()
    ret = dra[25:50:5]
    assert ret.shape[0] == 5


def test_slice_stop_and_step():
    dra = make_example_dynamicsizerecarray()
    ret = dra[:50:5]
    assert ret.shape[0] == 10


def test_idx_out_of_range():
    dra = make_example_dynamicsizerecarray()
    with pytest.raises(IndexError):
        _ = dra[200]


def test_slice_out_of_range():
    dra = make_example_dynamicsizerecarray()
    ret = dra[200:300]
    assert ret.shape[0] == 0


def test_slice_out_of_range():
    dra = make_example_dynamicsizerecarray()
    s = dra[10:20]
    for idx, val in enumerate(range(10, 20)):
        assert s["b"][idx] == val

    s = dra[10:2:20]
    for idx, val in enumerate(range(10, 2, 20)):
        assert s["b"][idx] == val


def test_full_range():
    dra = make_example_dynamicsizerecarray()
    nra = dra.to_recarray()

    assert dra.shape[0] == nra.shape[0]
    start = 0
    stop = nra.shape[0]

    dres = dra[start:stop]
    nres = nra[start:stop]

    assert len(dres) == len(nres)
