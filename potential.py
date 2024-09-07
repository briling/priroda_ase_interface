import subprocess
import numpy as np


class Potential():

    def make_input_file(self, atoms, inname, outname, mem=64, disk=-64, charge=0, mult=1):
        # TODO check the units. in init.xyz they are bohrs

        with open(inname, 'w') as f:
            print(f'''$system {mem=} {disk=} $end
$control
  theory=qm_n3
  basis=qm.in
  task=hessian
$end

$molecule
{charge=} {mult=}
cart
  {'\n  '.join(str(q) + ' ' + ' '.join(str(ri) for ri in r) for q, r in zip(atoms.numbers, atoms.positions))}
$end
            ''', file=f)


    def energy_and_forces(self, atoms, k, D, rEq):

        mem=64
        disk=-64
        charge=0
        mult=1
        inname='123.in'
        outname='123.out'

        self.make_input_file(atoms, inname, outname, mem=mem, disk=disk, charge=charge, mult=mult)

        # 1st option: sync run

        #subprocess.run(["p", inname, outname])
        #eng = subprocess.check_output(["grep", 'eng>', outname])
        output = subprocess.check_output(["p", inname])
        output = output.decode('ascii').split('\n')

        eng = [*reversed([*filter(lambda x: x.startswith('eng>'), reversed(output))])]

        energy = float(eng[1][7:])
        gradient = np.array([np.fromstring(line[7:], sep=' ', dtype=float) for line in eng[3:3+len(atoms)]])
        forces = -gradient
        # TODO check units
        # TODO add hessian

        #with open('ttt', 'w') as f:
        #    print(energy, file=f)
        #    print(gradient, file=f)
        #    for line in eng:
        #        print(line, file=f)

        # 2nd option: async
        # TODO

        return energy, forces
