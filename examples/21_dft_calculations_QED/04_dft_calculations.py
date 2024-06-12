import numpy as np
from matplotlib import pyplot as plt

from eminus import Atoms, SCF

polarization = np.array([1,1,1])/np.sqrt(3)
FREQ         = 0.1 # a.u.
A0_LIST      = [0.0, 0.01]#np.arange( 0.0, 12.0, 2.0 ) # a.u.
E            = np.zeros( len(A0_LIST) )

for A0i,A0 in enumerate( A0_LIST ):

    print(f"\n\n Working on A0 = {A0}")

    # # Start by creating an `Atoms` object for Argon
    # # Use a very small `ecut` for a fast calculation

    opt = {'pccg': 100}
    xc = 'lda,pw'
    pot = 'gth'
    guess = 'random'
    etol = 1e-8
    gradtol = 1e-7
    sic = False
    disp = False
    verbose = 2
    atoms = Atoms( 'Ar', [0, 0, 0], ecut=5, FREQ=FREQ, A0=A0, polarization=polarization )
    scf = SCF(atoms=atoms, xc=xc, pot=pot, guess=guess, etol=etol, gradtol=gradtol, opt=opt,
            sic=sic, disp=disp, verbose=verbose)

    # # Arguments for the minimizer can be passed through via the run function, e.g., for the conjugated-gradient form
    etot = scf.run(cgform=2)

    # # The total energy is a return value of the `SCF.run` function, but it is saved in the `SCF` object as well with all energy contributions
    print(f'\nSCF ENERGY = {etot} a.u.')
    E[A0i] = etot


plt.plot( A0_LIST, E - E[0], "-o" )
plt.xlabel("Coupling Strength, $A_0$ (a.u.)")
plt.ylabel("Ground State Energy (a.u.)")
plt.tight_layout()
plt.savefig("E.jpg", dpi=300)