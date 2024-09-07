
========================================
ASE_interface_force
========================================

The interface code (`interface.py`) communicates with Priroda
to use it as an ASE calculator for the force engine i-PI.

## Execution

To execute this, you have to:
```
$ i-pi input.xml
```
in one terminal, and in another:
```
python run.py
```

You have to be sure the `port` is the same in both files and that the `address` in `input.xml`
is the same as the `host` in `run.py`.

The input/output units are Å.

Use `input.xml` to define all the variables needed to run the dynamics using i-pi.

Use `init.xyz` to define the initial geometry and to define the cell size in the second line as a comment.

Use `clean_dir` to remove all the output and temporary files


## Installation

```
conda create --name ipi
conda activate ipi
conda install python=3.12
pip install ipi==3.0 ase==3.23
```
