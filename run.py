#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from ase.calculators.socketio import SocketClient
from ase import Atoms
import ase.io
from interface import PrirodaCalculator


tree = ET.parse('input.xml')
root = tree.getroot()
xyz = root.find('system').find('initialize').find('file').text.strip()
ffsocket = root.find('ffsocket')
host = ffsocket.find('address').text
port = int(ffsocket.find('port').text)
print(f'{xyz=}')
print(f'{host=} {port=}')

mol = ase.io.read(xyz)
mol.calc = PrirodaCalculator(charge=1, mult=2)

client = SocketClient(host=host, port=port)
client.run(mol)
