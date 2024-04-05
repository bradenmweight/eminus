#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
"""File input and output functionalities."""

from .cube import read_cube, write_cube
from .gth import read_gth
from .json import read_json, write_json
from .pdb import create_pdb_str, write_pdb
from .traj import read_traj, write_traj
from .xyz import read_xyz, write_xyz

__all__ = [
    'create_pdb_str',
    'read',
    'read_cube',
    'read_gth',
    'read_json',
    'read_traj',
    'read_xyz',
    'write',
    'write_cube',
    'write_json',
    'write_pdb',
    'write_traj',
    'write_xyz',
]


def read(filename, *args, **kwargs):
    """Unified file reader function.

    Args:
        filename: JSON input file path/name.
        *args: Pass-through arguments.
        **kwargs: Pass-through keyword arguments.
    """
    if filename.endswith('.json'):
        return read_json(filename, *args, **kwargs)
    if filename.endswith('.xyz'):
        return read_xyz(filename, *args, **kwargs)
    if filename.endswith(('.trj', '.traj')):
        return read_traj(filename, *args, **kwargs)
    if filename.endswith(('.cub', '.cube')):
        return read_cube(filename, *args, **kwargs)
    raise NotImplementedError('File ending not recognized.')


def write(obj, filename, *args, **kwargs):
    """Unified file writer function.

    Args:
        obj: Class object.
        filename: JSON input file path/name.
        *args: Pass-through arguments.
        **kwargs: Pass-through keyword arguments.
    """
    if filename.endswith('.json'):
        return write_json(obj, filename, *args, **kwargs)
    if filename.endswith('.xyz'):
        return write_xyz(obj, filename, *args, **kwargs)
    if filename.endswith(('.trj', '.traj')):
        return write_traj(obj, filename, *args, **kwargs)
    if filename.endswith(('.cub', '.cube')):
        return write_cube(obj, filename, *args, **kwargs)
    if filename.endswith('.pdb'):
        return write_pdb(obj, filename, *args, **kwargs)
    raise NotImplementedError('File ending not recognized.')
