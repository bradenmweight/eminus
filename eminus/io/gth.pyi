# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from numpy.typing import NDArray

def read_gth(
    atom: str,
    charge: int | None = ...,
    psp_path: str = ...,
) -> dict[str, int | float | NDArray[np.float64]]: ...
def mock_gth() -> dict[str, int | float | NDArray[np.float64]]: ...
