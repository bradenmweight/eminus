from eminus import Atoms, SCF, load_atoms, read_cube, read_xyz, save_atoms, write_cube
from eminus.units import bohr2ang

# Some file standards are supported to be read from
atom, X = read_xyz('CH4.xyz')

# To immediately create an Atoms object you can do the following
atoms = Atoms(*read_xyz('CH4.xyz'))

# Cube files are supported as well
# Here, lattice information are given as well
atom, X, Z, a, s = read_cube('CH4.cube')

# Create an Atoms object with it and start a DFT calculation
atoms = Atoms(atom=atom, X=X, a=a, s=s, verbose=2)
SCF(atoms)

# Write the total density to a cube file, e.g., to visualize it
write_cube(atoms, atoms.n, 'CH4_density')

# Please note that xyz files will use Angstrom as length units
# cube files have no standard, but atomic units will be assumed
# Units can always be converted using the unit conversion functionality
print(f'\nMethane coordinates in Bohr:\n{X}')
print(f'\nMethane coordinates in Angstrom:\n{bohr2ang(X)}')

# You can also save the Atoms object directly to load it later
save_atoms(atoms, 'CH4.pkl')
atoms = load_atoms('CH4.pkl')