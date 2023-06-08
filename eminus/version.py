#!/usr/bin/env python3
'''Package version number and version info function.'''
import importlib
import platform
import sys

__version__ = '2.4.0'
logo = (' ___ _____ _ ___ _ _ ___ \n'
        '| -_|     | |   | | |_ -|\n'
        '|___|_|_|_|_|_|_|___|___|\n')


def info():
    '''Print version numbers and availability of packages.'''
    dependencies = ('numpy', 'scipy')
    extras = ('torch', 'pyscf', 'plotly', 'nglview')
    dev = ('notebook', 'pylibxc', 'pytest', 'flake8', 'sphinx', 'furo')

    print(logo)
    print('--- Platform infos ---'
          f'\nPlatform  : {platform.system()} {platform.machine()}'
          f'\nRelease   : {platform.release()} {platform.version()}'
          '\n\n--- Version infos ---'
          f'\npython    : {sys.version.split()[0]}'
          f'\neminus    : {__version__}')
    for pkg in dependencies + extras + dev:
        try:
            module = importlib.import_module(pkg)
            try:
                print(f'{pkg:<10}: {module.__version__}')
            except AttributeError:
                # pylibxc does not use the standard version identifier
                print(f'{pkg:<10}: {module.version.__version__}')
        except ModuleNotFoundError:
            if pkg in dependencies:
                print(f'{pkg:<10}: Dependency not installed')
            elif pkg in extras:
                print(f'{pkg:<10}: Extra not installed')
    return


if __name__ == '__main__':
    info()
