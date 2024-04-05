# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from ..atoms import Atoms
from ..kpts import KPoints
from ..scf import SCF
from ..typing import Array1D, IntArray

def view(
    *args: Any,
    **kwargs: Any,
) -> Any: ...
def view_atoms(
    obj: Atoms | SCF,
    extra: NDArray[np.float64] | Sequence[NDArray[np.float64]] | None = ...,
    plot_n: bool = ...,
    percent: float = ...,
    surfaces: int = ...,
    size: IntArray = ...,
) -> Any: ...
def view_contour(
    obj: Atoms | SCF,
    field: NDArray[np.float64],
    axis: int = ...,
    value: float = ...,
    lines: int = ...,
    limits: Array1D = ...,
    zoom: float = ...,
    linewidth: float = ...,
    size: IntArray = ...,
) -> Any: ...
def view_file(
    filename: str | Sequence[str],
    isovalue: float = ...,
    gui: bool = ...,
    elec_symbols: Sequence[str] = ...,
    size: IntArray = ...,
    **kwargs: Any,
) -> Any: ...
def executed_in_notebook() -> bool: ...
def plot_bandstructure(
    scf: SCF,
    spin: int = ...,
    size: IntArray = ...,
) -> Any: ...
def view_kpts(
    kpts: KPoints,
    path: bool = ...,
    special: bool = ...,
    connect: bool = ...,
    size: IntArray = ...,
) -> Any: ...
