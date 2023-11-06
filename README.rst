###################
DynamicSizeRecarray
###################
|TestStatus| |PyPiStatus| |BlackStyle|

A dynamic, appandable version of Numpy's ``recarray``.

*******
install
*******

.. code:: bash

    pip install dynamicsizerecarray


*********
basic use
*********

.. code:: python

    import dynamicsizerecarray

    a = dynamicsizerecarray.DynamicSizeRecarray(
        dtype=[("hour", "u1"), ("minute", "u1"), ("temperature", "f8")]
    )

    a.append_record({"hour": 3, "minute": 53, "temperature": 22.434})

    print(len(a), a[0])


.. code:: bash

    1 (3, 53, 22.434)


When no more dynamic growth is needed, export to a numpy ``recarray``.

.. code:: python

    r = a.to_recarray()


*******
wording
*******

- ``record``: A ``dict`` with keys (and values) matching the ``dtype`` of
    the ``DynamicSizeRecarray``. This wording is adopted from ``pandas``.

- ``recarray``: Is short for ``np.core.records.recarray``.


.. |TestStatus| image:: https://github.com/cherenkov-plenoscope/dynamicsizerecarray/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/cherenkov-plenoscope/dynamicsizerecarray/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/dynamicsizerecarray
    :target: https://pypi.org/project/dynamicsizerecarray

.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

