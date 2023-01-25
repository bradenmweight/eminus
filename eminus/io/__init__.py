#!/usr/bin/env python3
'''File input and output functionalities.'''
from .cube import read_cube, write_cube
from .gth import read_gth
from .json import read_json, write_json
from .pdb import create_pdb_str, write_pdb
from .xyz import read_xyz, write_xyz
from ..logger import log

__all__ = ['create_pdb_str', 'read', 'read_cube', 'read_gth', 'read_json', 'read_xyz', 'write',
           'write_cube', 'write_json', 'write_pdb', 'write_xyz']


def read(*args, **kwargs):
    '''Unified file reader function.'''
    if args[0].endswith('.json'):
        return read_json(*args, **kwargs)
    elif args[0].endswith('.xyz'):
        return read_xyz(*args, **kwargs)
    elif args[0].endswith('.cube'):
        return read_cube(*args, **kwargs)
    else:
        log.error('No viable file ending found.')


def write(*args, **kwargs):
    '''Unified file writer function.'''
    if args[1].endswith('.json'):
        return write_json(*args, **kwargs)
    elif args[1].endswith('.xyz'):
        return write_xyz(*args, **kwargs)
    elif args[1].endswith('.cube'):
        return write_cube(*args, **kwargs)
    elif args[1].endswith('.pdb'):
        return write_pdb(*args, **kwargs)
    else:
        log.error('No viable file ending found.')