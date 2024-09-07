import tempfile
import subprocess
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes


class PrirodaCalculator(Calculator):

    implemented_properties = ["energy", "forces"]
    default_parameters = dict(mem=64, disk=-64, charge=0, mult=1, lr_cutoff=None)

    def __init__( self, restart=None, label=None, atoms=None, **kwargs):
        Calculator.__init__(self, restart, label, atoms, **kwargs)
        self.lr_cutoff = self.parameters.lr_cutoff  # ?
        self.N = 0
        self.positions = None
        self.pbc = np.array([False])
        self.cell = None
        self.cell_offsets = None
        tmp = tempfile.NamedTemporaryFile(suffix='.in', delete_on_close=False)
        self.inname = tmp.name
        tmp.close()
        print('parameters:', self.parameters)
        print('tmp input file:', self.inname)


    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self._update_neighborlists(atoms)
        energy, forces = self.energy_and_forces(atoms)
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


    def energy_and_forces(self, atoms):
        self.make_input_file(atoms)
        if True:
            eng = self.run_p_async()
        else:
            eng = self.run_p_sync()  # computes the whole hessian
        return self.parse_output(eng, len(atoms))


    def make_input_file(self, atoms):
        # TODO check the units. in init.xyz they are bohrs

        with open(self.inname, 'w') as f:
            print(f'''$system mem={self.parameters.mem} disk={self.parameters.disk} $end
$control
  theory=qm_n3
  basis=qm.in
  task=hessian
$end

$molecule
charge={self.parameters.charge} mult={self.parameters.mult}
cart
  {'\n  '.join(str(q) + ' ' + ' '.join(str(ri) for ri in r) for q, r in zip(atoms.numbers, atoms.positions))}
$end
            ''', file=f)


    def run_p_sync(self, inname):
        output = subprocess.check_output(["p", self.inname])
        output = output.decode('ascii').split('\n')
        return [*reversed([*filter(lambda x: x.startswith('eng>'), reversed(output))])]


    def run_p_async(self):
        with subprocess.Popen(["p", self.inname], bufsize=1,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL,
                              universal_newlines=True) as p:
            lines = []
            for line in p.stdout:
                if line.startswith('num>'):
                    lines.append(line.strip())
                    if line=='num>$end\n':
                        break
        p.wait()
        return lines


    def parse_output(self, eng, n):
        energy = float(eng[1][7:])
        gradient = np.array([np.fromstring(line[7:], sep=' ', dtype=float) for line in eng[3:3+n]])
        # TODO add hessian
        if False:
            with open('tmp', 'w') as f:
                print(energy, file=f)
                print(gradient, file=f)
                for line in eng:
                    print(line, file=f)
        return energy, -gradient
