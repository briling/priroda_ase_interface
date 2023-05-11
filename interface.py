import torch
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.calculators.calculator import Calculator, all_changes
from sklearn.neighbors import BallTree
#from .spookynet import SpookyNet
from potential import Potential 
#from .spookynet_ensemble import SpookyNetEnsemble

class myCalculator(Calculator):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #This module defines an ASE interface for Potential (class inside file potential.py which contains functions to compute 
    #energy and forces aimed to be used for MD).
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #implemented_properties = ["energy", "forces", "hessian", "dipole", "charges"]
    implemented_properties = ["energy", "forces"]

    default_parameters = dict(
        charge=0,
        magmom=0,
        #dtype=torch.float32,
        #use_gpu=True,
        lr_cutoff=None,
        #skin=0.3,  # skin-distance for building neighborlists
        #morse potential
        k=0.5,
        D=4.0,
        rEq=3.2,
    )

    def __init__(
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #To initialize something, check ignore_bad_restart_file in ase because is going to be removed from future versions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        self,
        restart=None,
        #ignore_bad_restart_file=False,
        label=None,
        atoms=None,
        **kwargs
    ):
        Calculator.__init__(
            #self, restart, ignore_bad_restart_file, label, atoms, **kwargs
            self, restart, label, atoms, **kwargs
        )
        self.lr_cutoff = self.parameters.lr_cutoff

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #This part might be necesary of loading parameters when using ML to make predictions
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #NOT LOADING ANY PARAMETR FROM EXTERNAL FILE
        #if type(self.parameters.load_from) is list:
        #    self.ensemble = True
        #    #self.spookynet = SpookyNetEnsemble(models=self.parameters.load_from)
        #    self.prueba = pruebaEnsemble(models=self.parameters.load_from)
        #    sr_cutoff = self.spookynet.models[0].cutoff
        #    self.cutoff = sr_cutoff
        #    self.use_lr = (
        #        self.spookynet.models[0].use_d4_dispersion
        #        or self.spookynet.models[0].use_electrostatics
        #    )
        #    for model in self.spookynet.models:
        #        assert sr_cutoff == model.cutoff
        #        assert self.use_lr == (
        #            model.use_d4_dispersion or model.use_electrostatics
        #        )
        #        if self.lr_cutoff is not None:  # overwrite lr_cutoff if one is given
        #            model.set_lr_cutoff(self.lr_cutoff)
        #        if model.lr_cutoff is not None:
        #            self.lr_cutoff = model.lr_cutoff
        #            self.cutoff = max(self.cutoff, model.lr_cutoff)
        #    if self.lr_cutoff is not None:
        #        for model in self.spookynet.models:
        #            model.set_lr_cutoff(self.lr_cutoff)
        #else:
        self.ensemble = False

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #Assign which class is going to compute the result values; in this case, the potential energy and forces
        #the preloaded parameters can be assigned here to the code used
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.mypotential = Potential()
        #self.spookynet = SpookyNet(load_from=self.parameters.load_from)

        #self.cutoff = self.spookynet.cutoff
        #self.use_lr = (
        #    self.spookynet.use_d4_dispersion or self.spookynet.use_electrostatics
        #)
        #if self.lr_cutoff is not None:  # overwrite lr_cutoff if one is given
        #    self.spookynet.set_lr_cutoff(self.lr_cutoff)
        #if self.spookynet.lr_cutoff is not None:
        #    self.lr_cutoff = self.spookynet.lr_cutoff
        #    self.cutoff = max(self.cutoff, self.spookynet.lr_cutoff)

        #self.dtype = self.parameters.dtype
        #self.spookynet.to(self.dtype).eval()

        #NOT USING GPUS
        # determine whether to use gpus
        #self.use_gpu = self.parameters.use_gpu and torch.cuda.is_available()
        #if self.use_gpu:
        #    self.spookynet.cuda()
        #self.calc_hessian = False
        #self.converged = True  # for compatibility with other calculators
        # for the neighborlist

        #self.skin2 = self.parameters.skin ** 2
        #assert self.parameters.skin >= 0
        #self.cutoff += (
        #    2 * self.parameters.skin
        #)  # cutoff needs to be larger when skin is used

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #Some basic asignations
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.N = 0
        self.positions = None
        self.pbc = np.array([False])
        self.cell = None
        self.cell_offsets = None

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Are all of this funcitons really needed for any other code?
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def _nsquared_neighborlist(self, atoms):
        pass
    #    if self.N != len(atoms):
    #        self.N = len(atoms)
    #        self.positions = np.copy(atoms.positions)
    #        self.pbc = np.array([False])
    #        self.cell = None
    #        self.cell_offsets = None
    #        idx = torch.arange(self.N, dtype=torch.int64)
    #        idx_i = idx.view(-1, 1).expand(-1, self.N).reshape(-1)
    #        idx_j = idx.view(1, -1).expand(self.N, -1).reshape(-1)
    #        # exclude self-interactions
    #        self.idx_i = idx_i[idx_i != idx_j]
    #        self.idx_j = idx_j[idx_i != idx_j]

    def _periodic_neighborlist(self, atoms):
        pass
    #    if (
    #        self.N != len(atoms)
    #        or (self.pbc != atoms.pbc).any()
    #        or (self.cell != atoms.cell).any()
    #        or ((self.positions - atoms.positions) ** 2).sum(-1).max() > self.skin2
    #    ):
    #        self.N = len(atoms)
    #        self.positions = np.copy(atoms.positions)
    #        self.pbc = atoms.pbc
    #        self.cell = atoms.cell
    #        idx_i, idx_j, cell_offsets = neighbor_list("ijS", atoms, self.cutoff)
    #        self.idx_i = torch.tensor(idx_i, dtype=torch.int64)
    #        self.idx_j = torch.tensor(idx_j, dtype=torch.int64)
    #        self.cell_offsets = torch.tensor(cell_offsets, dtype=self.dtype)

    def _non_periodic_neighborlist(self, atoms):
        pass
    #    if (
    #        self.N != len(atoms)
    #        or ((self.positions - atoms.positions) ** 2).sum(-1).max() >= self.skin2
    #    ):
    #        self.N = len(atoms)
    #        self.positions = np.copy(atoms.positions)
    #        self.pbc = np.array([False])
    #        self.cell = None
    #        self.cell_offsets = None
    #        tree = BallTree(self.positions)
    #        idx_i = []
    #        idx_j = tree.query_radius(self.positions, r=self.cutoff)
    #        for i in range(len(idx_j)):
    #            idx = idx_j[i]  # all neighbors with self-interaction
    #            idx = idx[idx != i]  # filter out self-interaction
    #            idx_i.append(np.full(idx.shape, i, idx.dtype))
    #            idx_j[i] = idx
    #        self.idx_i = torch.tensor(np.concatenate(idx_i), dtype=torch.int64)
    #        self.idx_j = torch.tensor(np.concatenate(idx_j), dtype=torch.int64)

    def _update_neighborlists(self, atoms):
        pass
    #    if atoms.pbc.any():
    #        self._periodic_neighborlist(atoms)
    #    else:
    #        if self.use_lr and self.lr_cutoff is None:
    #            self._nsquared_neighborlist(atoms)
    #        else:
    #            self._non_periodic_neighborlist(atoms)


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Energy calculator
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self._update_neighborlists(atoms)

        #args = {
        #    "Z": torch.tensor(atoms.numbers, dtype=torch.int64),
        #    "Q": torch.tensor([self.parameters.charge], dtype=self.dtype),
        #    "S": torch.tensor([self.parameters.magmom], dtype=self.dtype),
        #    "R": torch.tensor(atoms.positions, dtype=self.dtype, requires_grad=True),
        #    "idx_i": self.idx_i,
        #    "idx_j": self.idx_j,
        #    "cell": None
        #    if not atoms.pbc.any()
        #    else torch.tensor([atoms.cell], dtype=self.dtype),
        #    "cell_offsets": self.cell_offsets,
        #}

        # send args to GPU
        #if self.use_gpu:
        #    for key in args.keys():
        #        if isinstance(args[key], torch.Tensor):
        #            args[key] = args[key].cuda()

        #if self.calc_hessian:
        #    print("si")
        #    (
        #        energy,
        #        forces,
        #        hessian,
        #        f,
        #        ea,
        #        qa,
        #        ea_rep,
        #        ea_ele,
        #        ea_vdw,
        #        pa,
        #        c6,
        #    ) = self.spookynet.energy_and_forces_and_hessian(**args)
        #    # store hessian result
        #    if self.ensemble:
        #        self.results["hessian"] = hessian[0].detach().cpu().numpy()
        #        self.results["hessian_std"] = hessian[1].detach().cpu().numpy()
        #    else:
        #        self.results["hessian"] = hessian.detach().cpu().numpy()
        #else:
            #(
            #    energy,
            #    forces,
            #    f,
            #    ea,
            #    qa,
            #    ea_rep,
            #    ea_ele,
            #    ea_vdw,
            #    pa,
            #    c6,
        #if not self.calc_hessian:
        #dist=1.0
        #k=0.5
        #D=4.0
        #rEq=3.2

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #calculate the enrgy
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        (energy, forces) = self.mypotential.energy_and_forces(atoms.get_distance(1,0), self.parameters.k, self.parameters.D, self.parameters.rEq)
        #(energy, forces) = self.mypotential.morse(atoms.get_distance(1,0), self.parameters.k, self.parameters.D, self.parameters.rEq)
        #energy = self.mypotential.morse(self.get_distance(1,0), k, D, rEq)

        # store results
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #Results
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        self.results["energy"] = energy
        self.results["forces"] = forces
        #if self.ensemble:
        #    self.results["features"] = f[0].detach().cpu().numpy()
        #    self.results["features_std"] = f[1].detach().cpu().numpy()
        #    self.results["energy"] = energy[0].detach().cpu().item()
        #    self.results["energy_std"] = energy[1].detach().cpu().item()
        #    self.results["forces"] = forces[0].detach().cpu().numpy()
        #    self.results["forces_std"] = forces[1].detach().cpu().numpy()
        #    self.results["charges"] = qa[0].detach().cpu().numpy()
        #    self.results["charges_std"] = qa[1].detach().cpu().numpy()
        #    self.results["dipole"] = np.sum(
        #        atoms.get_positions() * self.results["charges"][:, None], 0
        #    )
        #    self.results["dipole_std"] = np.sum(
        #        atoms.get_positions() * self.results["charges_std"][:, None], 0
        #    )
        #else:
        #    self.results["features"] = f.detach().cpu().numpy()
        #    self.results["energy"] = energy.detach().cpu().item()
        #    self.results["forces"] = forces.detach().cpu().numpy()
        #    self.results["charges"] = qa.detach().cpu().numpy()
        #    self.results["dipole"] = np.sum(
        #        atoms.get_positions() * self.results["charges"][:, None], 0
        #    )

    def set_to_gradient_calculation(self):
        """ For compatibility with other calculators. """
        self.calc_hessian = False

    def set_to_hessian_calculation(self):
        """ For compatibility with other calculators. """
        self.calc_hessian = True

    def clear_restart_file(self):
        """ For compatibility with scripts that use file i/o calculators. """
        pass
