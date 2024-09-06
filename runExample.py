from ase.calculators.socketio import SocketClient
from ase import Atoms
from interface import PrirodaCalculator

molecule1 = Atoms('H2', positions=[(0.000,  0.000,  0.000),(0.000,  0.000,  5.500)  ])

molecule1.set_calculator(PrirodaCalculator())
#molecule1.set_calculator(myCalculator(rEq=4.0))

port = 3141
host = "localhost"
client = SocketClient(host=host, port=port)

client.run(molecule1)
