# Priroda ASE interface

The interface code (`interface.py`) communicates with [Priroda](http://rad.chem.msu.ru/~laikov)
to use it as an [ASE](https://wiki.fysik.dtu.dk/ase/) calculator for the force engine [i-PI](https://ipi-code.org/i-pi/).

## Installation

```
conda create --name ipi
conda activate ipi
conda install python=3.12
pip install ipi==3.0 ase==3.23
```

## Execution

To execute this, you have to run the engine in one terminal
```
conda activate ipi
i-pi input.xml
```
and the driver in another
```
conda activate ipi
./run.py
```

* `input.xml` defines all the variables needed to run the dynamics
   including the path to the initial geometry (here `init.xyz`)
* `init.xyz` defines the initial geometry and the cell size (keep the units Å)
