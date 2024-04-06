# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import float64
from numpy.typing import NDArray

def gga_c_chachiyo(
    n: NDArray[float64],
    dn_spin: NDArray[float64] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]: ...
def gga_c_chachiyo_spin(
    n: NDArray[float64],
    zeta: NDArray[float64],
    dn_spin: NDArray[float64] | None = ...,
    **kwargs: Any,
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]: ...
