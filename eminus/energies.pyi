# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from .atoms import Atoms
from .scf import SCF

@dataclasses.dataclass
class Energy:
    Ekin: float = ...
    Ecoul: float = ...
    Exc: float = ...
    Eloc: float = ...
    Enonloc: float = ...
    Eewald: float = ...
    Esic: float = ...
    Edisp: float = ...
    Eentropy: float = ...
    @property
    def Etot(self) -> float: ...
    def extrapolate(self) -> float: ...

def get_E(scf: SCF) -> float: ...
@overload
def get_Ekin(
    atoms: Atoms,
    Y: NDArray[np.complex128],
    ik: int,
) -> float: ...
@overload
def get_Ekin(
    atoms: Atoms,
    Y: list[NDArray[np.complex128]],
) -> float: ...
def get_Ecoul(
    atoms: Atoms,
    n: NDArray[np.float64],
    phi: NDArray[np.float64] | None = ...,
) -> float: ...
def get_Exc(
    scf: SCF,
    n: NDArray[np.float64],
    exc: NDArray[np.float64] | None = ...,
    n_spin: NDArray[np.float64] | None = ...,
    dn_spin: NDArray[np.float64] | None = ...,
    tau: NDArray[np.float64] | None = ...,
    Nspin: int = ...,
) -> float: ...
def get_Eloc(
    scf: SCF,
    n: NDArray[np.float64],
) -> float: ...
@overload
def get_Enonloc(
    scf: SCF,
    Y: NDArray[np.complex128],
    ik: int,
) -> float: ...
@overload
def get_Enonloc(
    scf: SCF,
    Y: list[NDArray[np.complex128]],
) -> float: ...
def get_Eewald(
    atoms: Atoms,
    gcut: float = ...,
    gamma: float = ...,
) -> float: ...
def get_Esic(
    scf: SCF,
    Y: list[NDArray[np.complex128]],
    n_single: NDArray[np.float64] | None = ...,
) -> float: ...
def get_Edisp(
    scf: SCF,
    version: str = ...,
    atm: bool = ...,
    xc: str | None = ...,
) -> float: ...
def get_Eband(
    scf: SCF,
    Y: list[NDArray[np.complex128]],
    **kwargs: Any,
) -> float: ...
def get_Eentropy(
    scf: SCF,
    epsilon: NDArray[np.float64],
    Efermi: float,
) -> float: ...