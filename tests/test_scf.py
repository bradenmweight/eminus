#!/usr/bin/env python3
"""Test the SCF class."""
import numpy as np
from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, RSCF, SCF, USCF
from eminus.tools import center_of_mass

atoms = Atoms('He', (0, 0, 0), ecut=2, unrestricted=True)


def test_atoms():
    """Test that the Atoms object is independent."""
    scf = SCF(atoms)
    assert id(scf.atoms) != id(atoms)


def test_xc():
    """Test that xc functionals are correctly parsed."""
    scf = SCF(atoms, xc='LDA,VWN')
    assert scf.xc == ['lda_x', 'lda_c_vwn']
    assert scf.xc_type == 'lda'

    scf.xc = 'PBE'
    assert scf.xc_type == 'gga'

    scf = SCF(atoms, xc=',')
    assert scf.xc == ['mock_xc', 'mock_xc']


def test_pot():
    """Test that potentials are correctly parsed and initialized."""
    scf = SCF(atoms, pot='GTH')
    assert scf.pot == 'gth'
    assert scf.psp == 'pade'
    assert hasattr(scf, 'gth')

    scf = SCF(atoms, pot='GTH', xc='pbe')
    assert scf.psp == 'pbe'

    scf.pot = 'test'
    assert scf.pot == 'gth'
    assert scf.psp == 'test'

    scf = SCF(atoms, pot='GE')
    assert scf.pot == 'ge'
    assert not hasattr(scf, 'gth')


def test_guess():
    """Test initialization of the guess method."""
    scf = SCF(atoms, guess='RAND')
    assert scf.guess == 'random'
    assert not scf.symmetric

    scf = SCF(atoms, guess='sym-pseudo')
    assert scf.guess == 'pseudo'
    assert scf.symmetric


def test_gradtol():
    """Test the convergence depending of the gradient norm."""
    etot = SCF(atoms, etol=1, gradtol=1e-2).run()
    assert etot < -1


def test_sic():
    """Test that the SIC routine runs."""
    scf = SCF(atoms, xc='pbe', opt={'sd': 1}, sic=True)
    scf.run()
    assert scf.energies.Esic != 0


@pytest.mark.parametrize('disp', [True, {'atm': False}])
def test_disp(disp):
    """Test that the dispersion correction routine runs."""
    pytest.importorskip('dftd3', reason='dftd3 not installed, skip tests')
    scf = SCF(atoms, opt={'sd': 1}, disp=disp)
    scf.run()
    assert scf.energies.Edisp != 0


def test_symmetric():
    """Test the symmetry option for H2 dissociation."""
    atoms = Atoms('H2', ((0, 0, 0), (0, 0, 6)), ecut=1, unrestricted=True)
    scf_symm = SCF(atoms, guess='symm-pseudo')
    scf_unsymm = SCF(atoms, guess='unsymm-pseudo')
    assert scf_symm.run() > scf_unsymm.run()


def test_opt():
    """Test the optimizer option."""
    atoms = Atoms('He', (0, 0, 0), ecut=1)
    scf = SCF(atoms, opt={'AUTO': 1})
    assert 'auto' in scf.opt
    scf.opt = {'sd': 1}
    assert 'sd' in scf.opt
    assert 'auto' not in scf.opt
    scf.run()
    assert 'sd' in scf._opt_log


def test_verbose():
    """Test the verbosity level."""
    scf = SCF(atoms)
    assert scf.verbose == atoms.verbose
    assert scf.log.verbose == atoms.log.verbose

    level = 'DEBUG'
    scf.verbose = level
    assert scf.verbose == level
    assert scf.log.verbose == level


def test_clear():
    """Test the clear function."""
    scf = SCF(atoms, opt={'sd': 1})
    scf.run()
    scf.clear()
    assert not scf.is_converged
    assert [x for x in (scf.Y, scf.n_spin, scf.dn_spin, scf.phi, scf.exc, scf.vxc,
            scf.vsigma, scf.vtau) if x is None]


@pytest.mark.parametrize('center', [None, np.diag(atoms.a) / 2])
def test_recenter(center):
    """Test the recenter function."""
    scf = SCF(atoms)
    scf.run()
    Vloc = scf.Vloc
    assert scf.is_converged

    scf.recenter(center)
    W = atoms.I(scf.W[0], 0)
    com = center_of_mass(scf.atoms.pos)
    # Check that the density is centered around the atom
    assert_allclose(center_of_mass(atoms.r, scf.n), com, atol=0.005)
    # Check that the orbitals are centered around the atom
    assert_allclose(center_of_mass(atoms.r, W[0, :, 0].conj() * W[0, :, 0]), com, atol=0.005)
    assert_allclose(center_of_mass(atoms.r, W[1, :, 0].conj() * W[1, :, 0]), com, atol=0.005)
    # Test that the local potential has been rebuild
    assert not np.array_equal(scf.Vloc, Vloc)
    assert scf.atoms.center == 'recentered'


def test_rscf():
    """Test the RSCF object."""
    scf = RSCF(atoms)
    assert not scf.atoms.unrestricted
    assert atoms.occ.Nspin == 2
    assert id(scf.atoms) != id(atoms)


def test_uscf():
    """Test the USCF object."""
    scf = USCF(atoms)
    assert scf.atoms.unrestricted
    assert atoms.occ.Nspin == 2
    assert id(scf.atoms) != id(atoms)


if __name__ == '__main__':
    import inspect
    import pathlib
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
