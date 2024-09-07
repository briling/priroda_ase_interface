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


    def run_p_sync(self, inname):
        output = subprocess.check_output(["p", inname])
        output = output.decode('ascii').split('\n')
        return [*reversed([*filter(lambda x: x.startswith('eng>'), reversed(output))])]


    def run_p_async(self, inname):
        with subprocess.Popen(["p", inname], bufsize=1,
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


    def energy_and_forces(self, atoms, k, D, rEq):

        mem=64
        disk=-64
        charge=0
        mult=1
        inname='123.in'
        outname='123.out'

        self.make_input_file(atoms, inname, outname, mem=mem, disk=disk, charge=charge, mult=mult)

        if True:
            eng = self.run_p_async(inname)
        else:
            eng = self.run_p_sync(inname) # computes the whole hessian

        return self.parse_output(eng, len(atoms))
