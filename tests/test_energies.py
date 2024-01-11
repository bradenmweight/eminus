#!/usr/bin/env python3
"""Test different energy contributions."""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, SCF
from eminus.energies import get_Eband

# The reference contributions are similar for the polarized and unpolarized case,
# but not necessary the same (for bad numerics)
E_ref = {
    'Ekin': 12.034539521,
    'Ecoul': 17.843659217,
    'Exc': -4.242215291,
    'Eloc': -58.537341024,
    'Enonloc': 8.152487157,
    'Eewald': -4.539675967,
    'Esic': -0.397817332,
    'Etot': -29.686363700
}

# Run the spin-unpaired calculation at first
atoms_unpol = Atoms('Ne', (0, 0, 0), ecut=10, unrestricted=False)
atoms_unpol.s = 20
scf_unpol = SCF(atoms_unpol, sic=True)
scf_unpol.run()
# Do the spin-paired calculation afterwards
# Use the orbitals from the restricted calculation as an initial guess for the unrestricted case
# This saves time and ensures we run into the same minimum
atoms_pol = Atoms('Ne', (0, 0, 0), ecut=10, unrestricted=True)
atoms_pol.s = 20
scf_pol = SCF(atoms_pol, sic=True)
scf_pol.W = [np.array([scf_unpol.W[0][0] / 2, scf_unpol.W[0][0] / 2])]
scf_pol.run()


@pytest.mark.parametrize('energy', E_ref.keys())
def test_energies_unpol(energy):
    """Check the spin-unpaired energy contributions."""
    E = getattr(scf_unpol.energies, energy)
    assert_allclose(E, E_ref[energy], atol=1e-4)


@pytest.mark.parametrize('energy', E_ref.keys())
def test_energies_pol(energy):
    """Check the spin-paired energy contributions."""
    E = getattr(scf_pol.energies, energy)
    assert_allclose(E, E_ref[energy], atol=1e-4)


def test_mgga_sic_unpol():
    """Check the spin-unpaired SIC energy for meta-GGAs."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    scf_unpol.xc = ':mgga_x_scan,:mgga_c_scan'
    scf_unpol.opt = {'auto': 1}
    scf_unpol.run()
    assert_allclose(scf_unpol.energies.Esic, -0.5374, atol=1e-4)


def test_mgga_sic_pol():
    """Check the spin-paired SIC energy for meta-GGAs."""
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    scf_pol.xc = ':mgga_x_scan,:mgga_c_scan'
    scf_pol.opt = {'auto': 1}
    scf_pol.run()
    assert_allclose(scf_pol.energies.Esic, -0.497, atol=1e-4)


def test_get_Eband_unpol():
    """Check the spin-unpaired band energy."""
    Eband = get_Eband(scf_unpol, scf_unpol.Y)
    assert_allclose(Eband, -290.8234, atol=1e-4)


def test_get_Eband_pol():
    """Check the spin-paired band energy."""
    Eband = get_Eband(scf_pol, scf_pol.Y, **scf_pol._precomputed)
    # About twice as large as the unpolarized case since we do not account for occupations
    # The "real" energy does not matter, we only want to minimize the band energy
    assert_allclose(Eband, -556.4278, atol=1e-4)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
