#!/usr/bin/env python3
'''Interface to use LibXC functionals.

For a list of available functionals, see: https://www.tddft.org/programs/libxc/functionals/
'''
try:
    from pylibxc import LibXCFunctional
except ImportError:
    print('ERROR: Necessary addon dependencies not found. To use this module,\n'
          '       install the package with addons, e.g., with "pip install eminus[addons]"')


def libxc_functional(exc, n, ret, spinpol):
    '''Handle LibXC exchange-correlation functionals.

    Only LDA functionals should be used.

    Reference: SoftwareX 7, 1.

    Args:
        exc (str | int): Exchange or correlation identifier.
        n (ndarray): Real-space electronic density.
        ret (str): Choose whether to return the energy density or the potential.
        spinpol (bool): Choose if a spin-polarized exchange-correlation functional will be used.

    Returns:
        ndarray: Exchange or correlation energy density or potential.
    '''
    if spinpol:
        print('WARNING: The LibXC routine will still use an unpolarized functional.')
    spin = 'unpolarized'

    inp = {'rho': n}
    # LibXC functionals have one integer and one string identifier
    try:
        func = LibXCFunctional(int(exc), spin)
    except ValueError:
        func = LibXCFunctional(exc, spin)
    out = func.compute(inp)
    if ret == 'density':
        return out['zk'].ravel()
    else:
        return out['vrho'].ravel()
