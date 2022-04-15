from eminus import Atoms, SCF
from eminus.addons import FLO, KSO
from eminus.energies import get_Esic

# Start by with a DFT calculation for neon
atoms = Atoms('Ne', [0, 0, 0])
SCF(atoms)

# Generate Kohn-Sham and Fermi-Löwdin orbitals
KSOs = KSO(atoms)
FLOs = FLO(atoms)

# Calculate the self-interaction energies
# The orbitals have to be in reciprocal space, so transform them
esic_kso = get_Esic(atoms, atoms.J(KSOs, False))
print(f'\nKSO-SIC energy = {esic_kso} Eh')

# The SIC energy will also be saved in the Atoms object
# The quality of the FLO-SIC energy will vary with the FOD guess
get_Esic(atoms, atoms.J(FLOs, False))
print(f'FLO-SIC energy = {atoms.energies.Esic} Eh')
print(f'\nAll energies:\n{atoms.energies}')
