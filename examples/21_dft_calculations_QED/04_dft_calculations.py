import numpy as np
from matplotlib import pyplot as plt

from eminus import Atoms, SCF, config, dft

config.threads = 2

polarization = np.array([1,1,1])/np.sqrt(3)
FREQ         = 0.01 # a.u.
A0_LIST      = np.arange( 0.0, 0.11, 0.01 ) # a.u.
E            = np.zeros( len(A0_LIST) )

psi = 0

for A0i,A0 in enumerate( A0_LIST ):

    print("\n\n Working on A0 = %1.3f" % A0)

    opt = None #{'pccg': 100}
    xc = 'lda,pw'
    pot = 'gth'
    guess = 'random'
    etol = 1e-12 # 1e-8
    gradtol = 1e-11 # 1e-7
    sic = False
    disp = False
    verbose = 4
    atoms = Atoms( ['H','H'], [[0,0,0],[0,0,0.8]], ecut=40, FREQ=FREQ, A0=A0, polarization=polarization )
    scf = SCF(atoms=atoms, xc=xc, pot=pot, guess=guess, etol=etol, gradtol=gradtol, opt=opt,
            sic=sic, disp=disp, verbose=verbose)
    if ( A0i >= 1 ):
        scf.W = psi

    etot = scf.run()
    psi = dft.get_psi(scf, scf.W)

    print(f'\nSCF ENERGY = {etot} a.u.')
    E[A0i] = etot


plt.plot( A0_LIST, E - E[0], "-o" )
plt.xlabel("Coupling Strength, $A_0$ (a.u.)", fontsize=15)
plt.ylabel("Ground State Energy (a.u.)", fontsize=15)
plt.tight_layout()
plt.savefig("E.jpg", dpi=300)