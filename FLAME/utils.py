import os
from ase.db import connect
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from ase import Atoms
from tqdm import tqdm
# from FLAME.schnetpack.data import AtomsData
from FLAME.dataprocess.flsf.scaffold import make_scf_tag,scaffold
import math

from ase.neighborlist import neighbor_list

root_path = os.path.dirname(os.path.normpath(__file__))
RDLogger.DisableLog('rdApp.*')
fluodb_df = pd.read_csv(f'{root_path}/dataprocess/FluoDB.csv')
def get_abs(row):
    idx, sdf = row
    dd1 = fluodb_df[fluodb_df['smiles'] == sdf['smiles']]
    dd1 = dd1[dd1['solvent']==sdf['solvent']]
    data = dd1['absorption/nm'].values[0]
    if data > 0:
        return data
    else:
        return 0

class BaseEnvironmentProvider:
    """
    Environment Providers are supposed to collect neighboring atoms within
    local, atom-centered environments. All environment providers should inherit
    from this class.

    """

    def get_environment(self, atoms):
        """
        Returns the neighbor indices and offsets

        Args:
            atoms (ase.Atoms): atomistic system

        Returns:
            neighborhood_idx (np.ndarray): indices of the neighbors with shape
                n_atoms x n_max_neighbors
            offset (np.ndarray): offset in lattice coordinates for periodic
                systems (otherwise zero matrix) of shape
                n_atoms x n_max_neighbors x 3

        """

        raise NotImplementedError

class SimpleEnvironmentProvider(BaseEnvironmentProvider):
    """
    A simple environment provider for small molecules where all atoms are each
    other's neighbors. It calculates full distance matrices and does not
    support cutoffs or periodic boundary conditions.
    """

    def get_environment(self, atoms, grid=None):
        n_atoms = atoms.get_global_number_of_atoms()

        if n_atoms == 1:
            neighborhood_idx = -np.ones((1, 1), dtype=np.dtype('float32'))
            offsets = np.zeros((n_atoms, 1, 3), dtype=np.dtype('float32'))
        else:
            neighborhood_idx = np.tile(
                np.arange(n_atoms, dtype=np.dtype('float32'))[np.newaxis], (n_atoms, 1)
            )

            neighborhood_idx = neighborhood_idx[
                ~np.eye(n_atoms, dtype=np.dtype('bool'))
            ].reshape(n_atoms, n_atoms - 1)

            if grid is not None:
                n_grid = grid.shape[0]
                neighborhood_idx = np.hstack([neighborhood_idx, -np.ones((n_atoms, 1))])
                grid_nbh = np.tile(
                    np.arange(n_atoms, dtype=np.dtype('float32'))[np.newaxis], (n_grid, 1)
                )
                neighborhood_idx = np.vstack([neighborhood_idx, grid_nbh])

            offsets = np.zeros(
                (neighborhood_idx.shape[0], neighborhood_idx.shape[1], 3),
                dtype=np.dtype('float32'),
            )
        return neighborhood_idx, offsets

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

def sol_smi2name(smis):
    gsol_df = pd.read_csv(f'{root_path}/dataprocess/gaussian_solvent_dict.csv')
    sol_dict = dict(list(zip(gsol_df['smiles'].values, gsol_df['name'].values)))
    res = []
    for smi in smis:
        if smi in sol_dict:
            res.append(sol_dict[smi])
        else:
            res.append('')
    return res

def valid_atoms(
    atoms,
    environment_provider=SimpleEnvironmentProvider(),
    collect_triples=False,
    centering_function=None,
):
    inputs = {}
    atoms.numbers.astype(np.dtype('int'))
    atoms.positions.astype(np.dtype('float32'))

    # get atom environment
    nbh_idx, offsets = environment_provider.get_environment(atoms)

    # Get neighbors and neighbor mask
    nbh_idx.astype(np.dtype('int'))
    # Get cells
    np.array(atoms.cell.array, dtype=np.dtype('float32'))
    offsets.astype(np.dtype('float32'))
    return True

def numpyfy_dict(data):
    """
    Transform floats, ints and dimensionless numpy in a dict to arrays to numpy arrays with dimenison.
    """
    for k, v in data.items():
        if type(v) in [int, float]:
            v = np.array([v])
        if v.shape == ():
            v = v[np.newaxis]
        data[k] = v
    return data

def get_center_of_mass(atoms):
    masses = atoms.get_masses()
#    print(atoms)
    return np.dot(masses, atoms.arrays["positions"]) / masses.sum()

def smiles2coord(smi, conf_num=3):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
#     smiles = Chem.MolToSmiles(mol)
    AllChem.Compute2DCoords(mol)
    coords = []
    for i in range(conf_num):
        AllChem.EmbedMolecule(mol)
        data = Chem.MolToXYZBlock(mol)
        coord = np.array([x[2:].strip().split() for x in data.strip().split('\n')[2:]]).astype(float)
        coords.append(coord)
    species = data.split()[1::4]
    return coords, species

def get_schnet_data(smiles, target, indices, save_path='', conf_num=1):
    if len(save_path) == 0:
        save_path = 'data/schnet/data.db'
    elif save_path[-2:] != 'db':
        save_path = save_path+'.db'
    available_properties = ["energy", "indices"]
    atoms_list = []
    property_list = []
    energies = target
    for idx,smi in tqdm(enumerate(smiles)):
        positions, species = smiles2coord(smi, conf_num=conf_num)
        species = ''.join(species)
        if len(species) == 0:
            print(f'invalid: ', smi)
            continue
        for i in range(conf_num):
#             valid_atoms(atm)
#             nbh_idx, offsets = environment.get_environment(atm)
#             ct = get_center_of_mass(atm)
            try:
                atm = Atoms(species, positions[i])
                energy = energies[idx]
                properties = {"energy": energy, "indices":indices[idx]}
                atoms_list.append(atm)
                property_list.append(properties)
            except:
                print(f'invalid: ', smi)
                break

    key_value_pairs_list = [dict() for _ in range(len(atoms_list))]
    
    if os.path.exists(save_path):
        os.remove(save_path)

    with connect(save_path) as conn:
        for at, prop, kv_pair in tqdm(zip(atoms_list, property_list, key_value_pairs_list)):
            data = {}
            # add available properties to database
            for pname in available_properties:
                try:
                    data[pname] = prop[pname]
                except:
                    raise Exception("Required property missing:" + pname)
            # transform to np.ndarray
            data = numpyfy_dict(data)
            conn.write(at, data=data, key_value_pairs=kv_pair)