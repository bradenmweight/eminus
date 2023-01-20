#!/usr/bin/env python3
'''Chachiyo LDA correlation.

Reference: J. Chem. Phys. 145, 021101.
'''
import numpy as np


def lda_c_chachiyo(n, **kwargs):
    '''Chachiyo parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_CHACHIYO and ID 287 in LibXC.
    Reference: J. Chem. Phys. 145, 021101.

    Args:
        n (ndarray): Real-space electronic density.

    Returns:
        tuple[ndarray, ndarray]: Chachiyo correlation energy density and potential.
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**third
    rs = pi34 / n**third

    a = -0.01554535  # (np.log(2) - 1) / (2 * np.pi**2)
    b = 20.4562557

    ec = a * np.log(1 + b / rs + b / rs**2)
    vc = ec + a * b * (2 + rs) / (3 * (b + b * rs + rs**2))
    return ec, np.array([vc])


def lda_c_chachiyo_spin(n, zeta, **kwargs):
    '''Chachiyo parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_CHACHIYO and ID 287 in LibXC.
    Reference: J. Chem. Phys. 145, 021101.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Returns:
        tuple[ndarray, ndarray]: Chachiyo correlation energy density and potential.
    '''
    third = 1 / 3
    pi34 = (3 / (4 * np.pi))**third
    rs = pi34 / n**third

    a0 = -0.01554535   # (np.log(2) - 1) / (2 * np.pi**2)
    a1 = -0.007772675  # (np.log(2) - 1) / (4 * np.pi**2)
    b0 = 20.4562557
    b1 = 27.4203609

    fzeta = ((1 + zeta)**(4 / 3) + (1 - zeta)**(4 / 3) - 2) / (2 * (2**third - 1))
    dfdzeta = (2 * (1 - zeta)**third - 2 * (1 + zeta)**third) / (3 - 3 * 2**third)

    ec0 = a0 * np.log(1 + b0 / rs + b0 / rs**2)
    ec1 = a1 * np.log(1 + b1 / rs + b1 / rs**2)
    ec = ec0 + (ec1 - ec0) * fzeta

    dec0drs = a0 / (1 + b0 / rs + b0 / rs**2) * b0 * (-1 / rs**2 - 2 / rs**3)
    dec1drs = a1 / (1 + b1 / rs + b1 / rs**2) * b1 * (-1 / rs**2 - 2 / rs**3)
    prefactor = ec - rs / 3 * (dec0drs + (dec1drs - dec0drs) * fzeta)
    vcup = prefactor + (ec1 - ec0) * dfdzeta * (1 - zeta)
    vcdw = prefactor - (ec1 - ec0) * dfdzeta * (1 + zeta)
    return ec, np.array([vcup, vcdw])
