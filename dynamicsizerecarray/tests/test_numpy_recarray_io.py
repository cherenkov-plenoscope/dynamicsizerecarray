import numpy as np
import tempfile
import os


def test_io():
    DTYPE = [("a", "i8"), ("b", "u2")]

    with tempfile.TemporaryDirectory(prefix="dynamicsizerecarray_") as tmp_dir:
        a = np.core.records.recarray(
            shape=18,
            dtype=DTYPE,
        )

        a_path = os.path.join(tmp_dir, "a.rec")
        with open(a_path, "wb") as f:
            f.write(a.tobytes())

        with open(a_path, "rb") as f:
            b = np.frombuffer(f.read(), dtype=DTYPE)

        for key in a.dtype.names:
            assert key in b.dtype.names
            assert a[key].dtype == b[key].dtype

        np.testing.assert_array_equal(a, b)
