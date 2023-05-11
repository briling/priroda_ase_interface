
========================================
ASE_interface_force
========================================


This is an example on how to make an interface file (`interface.py`) that comunicates a code (`potential.py`) that computes the
energy and forces of a system with ASE to convert it into an ASE calculator to further use it as the force engine for MD
calculations using i-PI. 

## Execution 

To execute this, you have to:
```
$ i-pi input.xml
```
in one terminal, and in another:
```
python runExample.py
```

## File description

With the conda environment `molDynSN`. You have to be sure the `port` is the same in both files and that the `adress` in `input.xml`
is the same as the `host` in `runExample.py`.

There must be a way t automatize it, but for now make sure the geometry in `init.xyz` is the same as the geometry in `runExample.py`.

Use `runExample.py` to initialize the geometry, define the calculator and pass the needed initial variables. 

Use `input.xml` to define all the variables needed to run the dynamics using i-pi.

Use `init.xyz` to define the initial geometry and to define the cell size in the second line as a comment.

`interface.py` contains the module that makes the interface between ASE and the given potential.

`potential.py` contains the code that computes the potential.
