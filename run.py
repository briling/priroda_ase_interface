#!/usr/bin/env python3

from ase.calculators.socketio import SocketClient
from ase import Atoms
import ase.io
from interface import PrirodaCalculator

mol = ase.io.read('init.xyz')
mol.calc = PrirodaCalculator(charge=0, spin=1)

port = 3141
host = "localhost"
client = SocketClient(host=host, port=port)
client.run(mol)
