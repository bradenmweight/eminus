#!/usr/bin/env python3
"""Test units conversion."""
import pytest

from eminus.units import (
    ang2bohr,
    bohr2ang,
    d2ebohr,
    ebohr2d,
    ev2ha,
    ha2ev,
    ha2kcalmol,
    ha2ry,
    kcalmol2ha,
    ry2ha,
)


@pytest.mark.parametrize('value', [-0.123, 0, 1.337])
def test_units(value):
    """Check that the identities are satisfied."""
    assert value == ha2ev(ev2ha(value))
    assert value == ha2kcalmol(kcalmol2ha(value))
    assert value == ha2ry(ry2ha(value))
    assert value == ang2bohr(bohr2ang(value))
    assert value == ebohr2d(d2ebohr(value))


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
