#!/usr/bin/env python3
"""SCF class definition."""
import copy
import logging
import time

import numpy as np

from .dft import get_epsilon, guess_pseudo, guess_random
from .energies import Energy, get_Edisp, get_Eewald, get_Esic
from .gth import GTH
from .logger import create_logger, get_level
from .minimizer import IMPLEMENTED as all_minimizer
from .potentials import IMPLEMENTED as all_potentials
from .potentials import init_pot
from .tools import center_of_mass, get_spin_squared
from .version import info
from .xc import parse_functionals, parse_xc_type


class SCF:
    """Perform direct minimizations.

    Args:
        atoms: Atoms object.

    Keyword Args:
        xc (str): Comma-separated exchange-correlation functional description.

            Adding 'libxc:' before a functional will use the Libxc interface for that functional,
            e.g., with :code:`libxc:mgga_x_scan,libxc:mgga_c_scan`.
        pot (str): Type of potential.

            Can be one of 'GTH' (default), 'Coulomb', 'Harmonic', or 'Ge'. Alternatively, a path to
            a directory containing custom GTH pseudopotential files can be given.
        guess (str): Initial guess for the wave functions.

            Can be one of 'random' (default) or 'pseudo'. Adding 'symm' to the string will use the
            same coefficients for both spin channels, e.g., :code:`symm-rand`.
        etol (float): Convergence tolerance of the total energy.
        gradtol (float): Convergence tolerance of the gradient norm.

            This tolerance will only be used in conjugate gradient methods.
        sic (bool): Calculate the Kohn-Sham Perdew-Zunger SIC energy at the end of the SCF step.
        disp (bool | dict): Calculate a dispersion correction.

            A dictionary can be used to pass arguments to the respective
            function, e.g., with :code:`{'version': 'd3zero', 'atm': False, 'xc': 'scan'}`.
        opt (dict | None): Dictionary to customize the minimization methods.

            The keys can be chosen out of 'sd', 'lm', 'pclm', 'cg', 'pccg', and 'auto'. Defaults to
            :code:`{'auto': 250}`.
        verbose (int | str): Level of output.

            Can be one of 'critical', 'error', 'warning', 'info' (default), or 'debug'. An integer
            value can be used as well, where larger numbers mean more output, starting from 0.
            Defaults to the verbosity level of the Atoms object.
    """
    def __init__(self, atoms, xc='lda,vwn', pot='gth', guess='random', etol=1e-7, gradtol=None,
                 sic=False, disp=False, opt=None, verbose=None):
        """Initialize the SCF object."""
        # Set opt here, better to not use mutable data types in signatures
        if opt is None:
            opt = {'auto': 250}

        # Set the input parameters
        self.atoms = atoms              #: Atoms object.
        self.log = create_logger(self)  #: Logger object.
        self.verbose = verbose          #: Verbosity level.
        self.xc = xc                    #: Exchange-correlation functional.
        self.pot = pot                  #: Used potential.
        self.guess = guess              #: Initial wave functions guess.
        self.etol = etol                #: Total energy convergence tolerance.
        self.gradtol = gradtol          #: Gradient norm convergence tolerance.
        self.sic = sic                  #: Enables the SIC energy calculation
        self.disp = disp                #: Enables the dispersion correction calculation.
        self.opt = opt                  #: Minimization methods.

        # Initialize the main attributes
        self.energies = Energy()        #: Energy object holding energy contributions.
        self.is_converged = False       #: Determines the SCF object convergence.

        # Listing of private attributes where the explicit setting of values is discouraged:
        # _xc_type, _psp, _symmetric, _opt_log, _gth

    @property
    def atoms(self):
        """Atoms object."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        # Build the atoms object if necessary and make a copy
        # This way the Atoms objects inside and outside the class are independent but both are build
        if not value.is_built:
            self._atoms = copy.copy(value.build())
        else:
            self._atoms = copy.copy(value)

    @property
    def xc(self):
        """Exchange-correlation functional."""
        return self._xc

    @xc.setter
    def xc(self, value):
        self._xc = parse_functionals(value.lower())
        # Determine the type of the functional combination
        self._xc_type = parse_xc_type(self._xc)
        if 'mock_xc' in self._xc:
            self.log.warning('Usage of mock functional detected.')

    @property
    def pot(self):
        """Used potential."""
        return self._pot

    @pot.setter
    def pot(self, value):
        if value.lower() in all_potentials:
            self._pot = value.lower()
            # Only set the pseudopotential type for GTH pseudopotentials
            if self._pot == 'gth':
                if 'gga' in self._xc_type:
                    self._psp = 'pbe'
                else:
                    self._psp = 'pade'
        # If pot is no supported potential treat it as a path to a directory containing GTH files
        else:
            self.log.info(f'Use the path "{value}" to search for GTH pseudopotential files.')
            self._psp = value
            self._pot = 'gth'
        # Build the potential
        if self._pot == 'gth':
            self._gth = GTH(self)
        self.Vloc = init_pot(self)

    @property
    def guess(self):
        """Initial wave functions guess."""
        return self._guess

    @guess.setter
    def guess(self, value):
        # Set the guess method
        value = value.lower()
        if 'rand' in value:
            self._guess = 'random'
        elif 'pseudo' in value:
            self._guess = 'pseudo'
        else:
            self.log.error(f'{value} is no valid initial guess.')
        # Check if a symmetric or unsymmetric guess is selected
        if 'sym' in value and 'unsym' not in value:
            self._symmetric = True
        else:
            self._symmetric = False

    @property
    def opt(self):
        """Minimization methods."""
        return self._opt

    @opt.setter
    def opt(self, value):
        # Set lowercase to all keys
        value = {k.lower(): v for k, v in value.items()}
        for opt in value:
            if opt not in all_minimizer:
                self.log.error(f'No minimizer found for "{opt}".')
        self._opt = value

    @property
    def verbose(self):
        """Verbosity level."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        # If no verbosity is given use the one from the Atoms object
        if value is None:
            value = self.atoms.verbose
        self._verbose = get_level(value)
        self.log.verbose = self._verbose

    def run(self, **kwargs):
        """Run the self-consistent field (SCF) calculation."""
        # Print some information about the calculation
        if self.log.level <= logging.DEBUG:
            info()
        self.log.debug(f'\n--- System information ---\n{self.atoms}\n'
                       f'Spin handling: {"un" if self.atoms.Nspin == 2 else ""}restricted\n'
                       f'Number of states: {self.atoms.Nstate}\n'
                       f'Occupation per state:\n{self.atoms.f}\n'
                       f'\n--- Cell information ---\nSide lengths: {self.atoms.a} a0\n'
                       f'Sampling per axis: {self.atoms.s}\n'
                       f'Cut-off energy: {self.atoms.ecut} Eh\n'
                       f'Compression: {len(self.atoms.G2c) / len(self.atoms.G2):.5f}\n'
                       f'\n--- Calculation information ---\n{self}\n\n--- SCF data ---')

        # Calculate Ewald energy that only depends on the system geometry
        self.energies.Eewald = get_Eewald(self.atoms)
        # Build the initial wave function as long as W there is no W
        if not hasattr(self, 'W'):
            if 'random' in self.guess:
                self.W = guess_random(self, symmetric=self._symmetric)
            elif 'pseudo' in self.guess:
                self.W = guess_pseudo(self, symmetric=self._symmetric)

        # Start the minimization procedures
        self.clear()
        Etots = []
        for imin in self.opt:
            # Call the minimizer
            self.log.info(f'Start {all_minimizer[imin].__name__}...')
            start = time.perf_counter()
            Elist = all_minimizer[imin](self, self.opt[imin], **kwargs)
            end = time.perf_counter()
            # Save the minimizer results
            self._opt_log[imin] = {}
            self._opt_log[imin]['iter'] = len(Elist)
            self._opt_log[imin]['time'] = end - start
            Etots += Elist
            # Do not start other minimizations if one converged
            if self.is_converged:
                break
        if self.is_converged:
            self.log.info(f'SCF converged after {len(Etots)} iterations.')
        else:
            self.log.warning('SCF not converged!')

        # Calculate SIC energy if desired
        if self.sic:
            self.energies.Esic = get_Esic(self, self.Y)
        # Calculate dispersion correction energy if desired
        if isinstance(self.disp, dict):
            self.energies.Edisp = get_Edisp(self, **self.disp)
        elif self.disp:
            self.energies.Edisp = get_Edisp(self)

        # Print minimizer timings
        self.log.debug('\n--- SCF results ---')
        t_tot = 0
        for imin in self._opt_log:
            N = self._opt_log[imin]['iter']
            t = self._opt_log[imin]['time']
            t_tot += t
            self.log.debug(f'Minimizer: {imin}'
                           f'\nIterations: {N}'
                           f'\nTime: {t:.5f} s'
                           f'\nTime/Iteration: {t / N:.5f} s')
        self.log.info(f'Total SCF time: {t_tot:.5f} s')
        # Print the S^2 expectation value for unrestricted calculations
        if self.atoms.Nspin == 2:
            self.log.info(f'<S^2> = {get_spin_squared(self):.6e}')
        # Print energy data
        if self.log.level <= logging.DEBUG:
            self.log.debug('\n--- Energy data ---\n'
                           f'Eigenenergies:\n{get_epsilon(self, self.W)}\n\n{self.energies}')
        else:
            self.log.info(f'Etot = {self.energies.Etot:.9f} Eh')
        return self.energies.Etot

    kernel = run

    def recenter(self, center=None):
        """Recenter the system inside the cell.

        Keyword Args:
            center (float | list | tuple | ndarray | None): Point to center the system around.
        """
        atoms = self.atoms
        # Get the COM before centering the atoms
        com = center_of_mass(atoms.X)
        # Run the recenter method of the atoms object
        self.atoms.recenter(center=center)
        if center is None:
            dr = com - atoms.a / 2
        else:
            center = np.asarray(center)
            dr = com - center

        # Shift orbitals and density
        if hasattr(self, 'W'):
            self.W = atoms.T(self.W, dr=-dr)
        # Transform the density to the reciprocal space, shift, and transform back
        if hasattr(self, 'n'):
            Jn = atoms.J(self.n)
            TJn = atoms.T(Jn, dr=-dr)
            self.n = np.real(atoms.I(TJn))

        # Recalculate the pseudopotential since it depends on the structure factor
        self.pot = self.pot
        # Clear intermediate results to make sure no one uses the unshifted results
        self.clear()
        return self

    def clear(self):
        """Initialize or clear intermediate results."""
        self.Y = None           # Orthogonal wave functions
        self.n_spin = None      # Electronic densities per spin
        self.dn_spin = None     # Gradient of electronic densities per spin
        self.tau = None         # Kinetic energy densities per spin
        self.phi = None         # Hartree field
        self.exc = None         # Exchange-correlation energy density
        self.vxc = None         # Exchange-correlation potential
        self.vsigma = None      # n times d exc/d |dn|^2
        self.vtau = None        # d exc/d tau
        self._precomputed = {}  # Dictionary of pre-computed values not to be saved
        self._opt_log = {}      # Log of the optimization procedure
        return self

    def __repr__(self):
        """Print the most important parameters stored in the SCF object."""
        # Use chr(10) to create a linebreak since backslashes are not allowed in f-strings
        return f'XC functionals: {self.xc}\n' \
               f'Potential: {self.pot}\n' \
               f'{f"GTH files: {self._psp}" + chr(10) if self.pot == "gth" else ""}' \
               f'Starting guess: {self.guess}\n' \
               f'Symmetric guess: {self._symmetric}\n' \
               f'Energy convergence tolerance: {self.etol} Eh\n' \
               f'Gradient convergence tolerance: {self.gradtol}\n' \
               f'Non-local potential: {self._gth.NbetaNL > 0 if self.pot == "gth" else "false"}\n' \
               f'Converged: {self.is_converged}'

class RSCF(SCF):
    """SCF class for spin-paired systems.

    Inherited from :class:`eminus.scf.SCF`.

    In difference to the SCF class, this class will not build the original Atoms object, only the
    one attributed to the class.
    """
    @property
    def atoms(self):
        """Atoms object."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        self._atoms = copy.copy(value)
        self._atoms._set_states(Nspin=1)
        if not self._atoms.is_built:
            self._atoms = self._atoms.build()


class USCF(SCF):
    """SCF class for spin-polarized systems.

    Inherited from :class:`eminus.scf.SCF`.

    In difference to the SCF class, this class will not build the original Atoms object, only the
    one attributed to the class.
    """
    @property
    def atoms(self):
        """Atoms object."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        self._atoms = copy.copy(value)
        self._atoms._set_states(Nspin=2)
        if not self._atoms.is_built:
            self._atoms = self._atoms.build()
