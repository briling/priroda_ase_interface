import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from sklearn.neighbors import BallTree
from potential import Potential

class PrirodaCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    default_parameters = dict(
        charge=0,
        spin=0,
        lr_cutoff=None,
        k=0.5,
        D=4.0,
        rEq=3.2,
    )

    def __init__(
        self,
        restart=None,
        #ignore_bad_restart_file=False,
        label=None,
        atoms=None,
        **kwargs
    ):
        Calculator.__init__(self, restart, label, atoms, **kwargs)
        self.lr_cutoff = self.parameters.lr_cutoff
        self.ensemble = False   ######### TODO not needed?

        self.mypotential = Potential() ####################################################################

        self.N = 0
        self.positions = None
        self.pbc = np.array([False])
        self.cell = None
        self.cell_offsets = None


    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self._update_neighborlists(atoms)
        (energy, forces) = self.mypotential.energy_and_forces(atoms, self.parameters.k, self.parameters.D, self.parameters.rEq)
        self.results["energy"] = energy
        self.results["forces"] = forces


    def set_to_gradient_calculation(self):
        """ For compatibility with other calculators. """
        self.calc_hessian = False


    def set_to_hessian_calculation(self):
        """ For compatibility with other calculators. """
        self.calc_hessian = True


    def clear_restart_file(self):
        """ For compatibility with scripts that use file i/o calculators. """
        pass

    def _nsquared_neighborlist(self, atoms):
        pass

    def _periodic_neighborlist(self, atoms):
        pass

    def _non_periodic_neighborlist(self, atoms):
        pass

    def _update_neighborlists(self, atoms):
        pass
