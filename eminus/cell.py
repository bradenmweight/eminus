#!/usr/bin/env python3
"""Cell wrapper function."""
import numpy as np

from .atoms import Atoms
from .data import LATTICE_VECTORS
from .utils import molecule2list

#: Crystal structures with their respective lattice and basis.
STRUCTURES = {
    'sc': {
        'lattice': 'sc',
        'basis': [[0, 0, 0]]
    },
    'fcc': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0]]
    },
    'bcc': {
        'lattice': 'bcc',
        'basis': [[0, 0, 0]]
    },
    'tetragonal': {
        'lattice': 'sc',
        'basis': [[0, 0, 0]]
    },
    'orthorhombic': {
        'lattice': 'sc',
        'basis': [[0, 0, 0]]
    },
    'diamond': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]]
    },
    'zincblende': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]]
    },
    'rocksalt': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0], [1 / 2, 0, 0]]
    },
    'cesiumchloride': {
        'lattice': 'sc',
        'basis': [[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]]
    },
    'fluorite': {
        'lattice': 'fcc',
        'basis': [[0, 0, 0], [1 / 4, 1 / 4, 1 / 4], [3 / 4, 3 / 4, 3 / 4]]
    }
}


def Cell(atom, lattice, ecut, a, basis=None, verbose=None):
    """Wrapper to create Atoms classes for crytal systems.

    Args:
        atom (str | list | tuple): Atom symbols.
        lattice (str | list | tuple | ndarray): Lattice system.
        ecut (float):  Cut-off energy.
        a (float | list | tuple | ndarray | None): Cell size.

    Keyword Args:
        basis (list | tuple | ndarray | None): Lattice basis.
        verbose (int | str | None): Level of output.

    Returns:
        Atoms object.
    """
    # Get the lattice vectors from a string or use them directly
    if isinstance(lattice, str):
        if basis is None:
            basis = STRUCTURES[lattice]['basis']
        lattice = STRUCTURES[lattice]['lattice']
        lattice_vectors = np.asarray(LATTICE_VECTORS[lattice])
    else:
        if basis is None:
            basis = [[0, 0, 0]]
        lattice_vectors = np.asarray(lattice)

    # Only scale the lattice vectors with if a is given
    if a is not None:
        lattice_vectors = a * lattice_vectors
        basis = a * np.asarray(basis)

    # Account for different atom and basis sizes
    if isinstance(atom, str):
        atom_list = molecule2list(atom)
    else:
        atom_list = atom
    if len(atom_list) != len(basis):
        atom = [atom] * len(basis)
    return Atoms(atom, basis, ecut=ecut, a=lattice_vectors, verbose=verbose)
