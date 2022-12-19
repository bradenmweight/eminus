#!/usr/bin/env python3
'''Test total energies for a small set of spin-paired systems.'''
import inspect
import pathlib

from numpy.testing import assert_allclose
import pytest

from eminus import Atoms, read_xyz, RSCF

# Total energies from a spin-polarized calculation with PWDFT.jl with same parameters as below
# Closed-shell systems have the same energy for spin-paired and -polarized calculations
E_ref = {
    'H2': -1.103621,
    'He': -2.542731,
    'LiH': -0.793497,
    'CH4': -7.699509,
    'Ne': -29.876365
}


@pytest.mark.parametrize('system', E_ref.keys())
def test_unpolarized(system):
    '''Compare total energies for a test system with a reference value (spin-paired).'''
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
    a = 10
    ecut = 10
    s = 30
    xc = 'lda,vwn'
    guess = 'random'
    etol = 1e-6
    min = {'sd': 3, 'pccg': 18}

    atom, X = read_xyz(str(file_path.joinpath(f'{system}.xyz')))
    atoms = Atoms(atom, X, a=a, ecut=ecut, s=s, verbose='warning')
    E = RSCF(atoms, xc=xc, guess=guess, etol=etol, min=min).run()
    assert_allclose(E, E_ref[system], atol=etol)
    return


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
