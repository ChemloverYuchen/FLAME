import threading
from collections import OrderedDict
from random import Random
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from rdkit import Chem

from .scaler import StandardScaler
from FLAME.dataprocess.abtmpnn.features import get_features_generator
from FLAME.dataprocess.abtmpnn.features import BatchMolGraph, MolGraph


# Cache of graph featurizations
CACHE_GRAPH = False
SMILES_TO_GRAPH: Dict[str, MolGraph] = {}

# A cache is a reserved storage location that collects temporary data to help websites, browsers, and apps load faster.


def cache_graph() -> bool:
    r"""Returns whether :class:`~abtmpnn.features.MolGraph`\ s will be cached."""
    return CACHE_GRAPH


def set_cache_graph(cache_graph: bool) -> None:
    r"""Sets whether :class:`~abtmpnn.features.MolGraph`\ s will be cached."""
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph


# Cache of RDKit molecules
CACHE_MOL = True
SMILES_TO_MOL: Dict[str, Chem.Mol] = {}


def cache_mol() -> bool:
    r"""Returns whether RDKit molecules will be cached."""
    return CACHE_MOL


def set_cache_mol(cache_mol: bool) -> None:
    r"""Sets whether RDKit molecules will be cached."""
    global CACHE_MOL
    CACHE_MOL = cache_mol


class MoleculeDatapoint:
    """A :class:`MoleculeDatapoint` contains a single molecule and its associated features and targets."""

    def __init__(self,
                 smiles: List[str],
                 targets: List[Optional[float]] = None,
                 row: OrderedDict = None,
                 features: np.ndarray = None,
                 features_generator: List[str] = None,
                 mol_adj: np.ndarray = None,
                 mol_dist: np.ndarray = None,
                 mol_clb: np.ndarray = None):
        """
        (all the molecules)
        :param smiles: A list of the SMILES strings for the molecules.
        :param targets: A list of targets for the molecule (contains None for unknown target values).
        :param row: The raw CSV row containing the information for this molecule.
        :param features: A numpy array containing additional features (e.g., Morgan fingerprint).
        :param features_generator: A list of features generators to use.
        :param mol_adj: A numpy array containing additional atom matrix (adjacency) to featurize the molecule
        :param mol_dist: A numpy array containing additional atom matrix (distance) to featurize the molecule
        :param mol_clb: A numpy array containing additional atom matrix (coulomb) to featurize the molecule

        """
        if features is not None and features_generator is not None:
            raise ValueError(
                'Cannot provide both loaded features and a features generator.')

        self.smiles = smiles
        self.targets = targets
        self.row = row
        self.features = features
        self.features_generator = features_generator

        self.mol_adj = mol_adj
        self.mol_dist = mol_dist
        self.mol_clb = mol_clb

        # Generate additional features if given a generator
        if self.features_generator is not None:
            self.features = []

            for fg in self.features_generator:
                # abtmpnn/features/features_generators.py
                features_generator = get_features_generator(fg)
                # defined in mol(self), which contains a list of molecules (in rdkit.chem format)
                for m in self.mol:
                    if m is not None and m.GetNumHeavyAtoms() > 0:  # m.GetNumHeavyAtoms(): inbuilt rdkit.chem property
                        # returns our number of heavy atoms (atomic number > 1)
                        # compute descriptors that are in the len of non-H atoms
                        self.features.extend(features_generator(m))
                    # for H2
                    elif m is not None and m.GetNumHeavyAtoms() == 0:
                        # not all features are equally long, so use methane as dummy molecule to determine length
                        self.features.extend(
                            np.zeros(len(features_generator(Chem.MolFromSmiles('C')))))

            self.features = np.array(self.features)

        # Fix nans in features
        replace_token = 0
        if self.features is not None:
            self.features = np.where(
                np.isnan(self.features), replace_token, self.features)

        # Fix nans in mol_adj
        if self.mol_adj is not None:
            self.mol_adj = np.where(
                np.isnan(self.mol_adj), replace_token, self.mol_adj)
            self.mol_adj = np.where(
                np.isinf(self.mol_adj), replace_token, self.mol_adj)

        # Fix nans in mol_dist
        if self.mol_dist is not None:
            self.mol_dist = np.where(
                np.isnan(self.mol_dist), replace_token, self.mol_dist)
            self.mol_dist = np.where(
                np.isinf(self.mol_dist), replace_token, self.mol_dist)

        # Fix nans in mol_clb
        if self.mol_clb is not None:
            self.mol_clb = np.where(
                np.isnan(self.mol_clb), replace_token, self.mol_clb)
            self.mol_clb = np.where(
                np.isinf(self.mol_clb), replace_token, self.mol_clb)

        # Save a copy of the raw features and targets to enable different scaling later on
        self.raw_features, self.raw_targets = self.features, self.targets
        self.raw_mol_adj, self.raw_mol_dist, self.raw_mol_clb = \
            self.mol_adj, self.mol_dist, self.mol_clb

    @property
    # A list of molecules in rdkit.chem format. (converted from SMILES list)
    def mol(self) -> List[Chem.Mol]:
        """Gets the corresponding list of RDKit molecules for the corresponding SMILES list."""
        mol = [SMILES_TO_MOL.get(s, Chem.MolFromSmiles(s))
               for s in self.smiles]

        if cache_mol():  # zip(): Join two tuples together
            for s, m in zip(self.smiles, mol):
                SMILES_TO_MOL[s] = m

        return mol

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in the :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return len(self.smiles)

    def set_features(self, features: np.ndarray) -> None:
        """
        Sets the features of the molecule.

        :param features: A 1D numpy array of features for the molecule.
        """
        self.features = features

    def set_mol_adj(self, mol_adj: np.ndarray) -> None:
        """
        Sets the adjacency matrix of the molecule.

        :param mol_adj: A 2D numpy array of features for the molecule.
        """
        self.mol_adj = mol_adj

    def set_mol_dist(self, mol_dist: np.ndarray) -> None:
        """
        Sets the distance matrix of the molecule.

        :param mol_dist: A 2D numpy array of features for the molecule.
        """
        self.mol_dist = mol_dist

    def set_mol_clb(self, mol_clb: np.ndarray) -> None:
        """
        Sets the coulomb matrix of the molecule.

        :param mol_clb: A 2D numpy array of features for the molecule.
        """
        self.mol_clb = mol_clb

    def extend_features(self, features: np.ndarray) -> None:
        """
        Extends the features of the molecule.

        :param features: A 1D numpy array of extra features for the molecule.
        """
        self.features = np.append(
            self.features, features) if self.features is not None else features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[Optional[float]]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        self.features, self.targets = self.raw_features, self.raw_targets
        self.mol_adj, self.mol_dist, self.mol_clb = \
            self.raw_mol_adj, self.raw_mol_dist, self.raw_mol_clb


class MoleculeDataset(Dataset):
    r"""A :class:`MoleculeDataset` contains a list of :class:`MoleculeDatapoint`\ s with access to their attributes."""

    def __init__(self, data: List[MoleculeDatapoint]):
        r"""
        :param data: A list of :class:`MoleculeDatapoint`\ s.
        """
        self._data = data  # A list of MoleculeDatapoint s.
        #parameters: smiles; target; row; features;
        #features_generator; atom_descriptors; bond_features; overwrite_default_atom_features; overwrite_default_bond_features
        self._scaler = None
        self._batch_graph = None
        self._random = Random()

    def smiles(self, flatten: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Returns a list containing the SMILES list associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned SMILES to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of SMILES, depending on :code:`flatten`.
        """
        if flatten:
            return [smiles for d in self._data for smiles in d.smiles]

        return [d.smiles for d in self._data]

    def mols(self, flatten: bool = False) -> Union[List[Chem.Mol], List[List[Chem.Mol]]]:
        """
        Returns a list of the RDKit molecules associated with each :class:`MoleculeDatapoint`.

        :param flatten: Whether to flatten the returned RDKit molecules to a list instead of a list of lists.
        :return: A list of SMILES or a list of lists of RDKit molecules, depending on :code:`flatten`.
        """
        if flatten:
            return [mol for d in self._data for mol in d.mol]

        return [d.mol for d in self._data]

    @property
    def number_of_molecules(self) -> int:
        """
        Gets the number of molecules in each :class:`MoleculeDatapoint`.

        :return: The number of molecules.
        """
        return self._data[0].number_of_molecules if len(self._data) > 0 else None

    def batch_graph(self) -> List[BatchMolGraph]:
        r"""
        Constructs a :class:`~abtmpnn.features.BatchMolGraph` with the graph featurization of all the molecules.

        .. note::
           The :class:`~abtmpnn.features.BatchMolGraph` is cached in after the first time it is computed
           and is simply accessed upon subsequent calls to :meth:`batch_graph`. This means that if the underlying
           set of :class:`MoleculeDatapoint`\ s changes, then the returned :class:`~abtmpnn.features.BatchMolGraph`
           will be incorrect for the underlying data.

        :return: A list of :class:`~abtmpnn.features.BatchMolGraph` containing the graph featurization of all the
                 molecules in each :class:`MoleculeDatapoint`.
        """
        if self._batch_graph is None:
            self._batch_graph = []

            mol_graphs = []
            for d in self._data:
                mol_graphs_list = []
                # iterate through all the molecules and generate their graphs
                for s, m in zip(d.smiles, d.mol):
                    if s in SMILES_TO_GRAPH:
                        mol_graph = SMILES_TO_GRAPH[s]
                    else:
                        mol_graph = MolGraph(
                            m, d.mol_adj, d.mol_dist, d.mol_clb)
                        if cache_graph():
                            SMILES_TO_GRAPH[s] = mol_graph
                    mol_graphs_list.append(mol_graph)
                mol_graphs.append(mol_graphs_list)

            # class BatchMolGraph: abtmpnn/features/featurization.py
            self._batch_graph = [BatchMolGraph(
                [g[i] for g in mol_graphs]) for i in range(len(mol_graphs[0]))]
        # the graph featurization of all the molecules.
        return self._batch_graph

    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].features is None:
            return None

        # additional features for all the molecules
        return [d.features for d in self._data]

    def adj_features(self) -> List[np.ndarray]:
        """
        Returns the adjacency matrix associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the adjacency matrix
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].mol_adj is None:
            return None

        return [d.mol_adj for d in self._data]

    def dist_features(self) -> List[np.ndarray]:
        """
        Returns the distance matrix associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the distance matrix
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].mol_dist is None:
            return None

        return [d.mol_dist for d in self._data]

    def clb_features(self) -> List[np.ndarray]:
        """
        Returns the coulomb matrix associated with each molecule (if they exit).

        :return: A list of 2D numpy arrays containing the coulomb matrix
                 for each molecule or None if there are no features.
        """
        if len(self._data) == 0 or self._data[0].mol_clb is None:
            return None

        return [d.mol_clb for d in self._data]

    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        return [d.targets for d in self._data]

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self._data[0].num_tasks() if len(self._data) > 0 else None

    def features_size(self) -> int:
        """
        Returns the size of the additional features vector associated with the molecules.

        :return: The size of the additional features vector.
        """
        return len(self._data[0].features) if len(self._data) > 0 and self._data[0].features is not None else None

    # def atom_descriptors_size(self) -> int:
    #     """
    #     Returns the size of custom additional atom descriptors vector associated with the molecules.
    #
    #     :return: The size of the additional atom descriptor vector.
    #     """
    #     return len(self._data[0].atom_descriptors[0]) \
    #         if len(self._data) > 0 and self._data[0].atom_descriptors is not None else None
    #
    # def atom_features_size(self) -> int:
    #     """
    #     Returns the size of custom additional atom features vector associated with the molecules.
    #
    #     :return: The size of the additional atom feature vector.
    #     """
    #     return len(self._data[0].atom_features[0]) \
    #         if len(self._data) > 0 and self._data[0].atom_features is not None else None
    #
    # def bond_features_size(self) -> int:
    #     """
    #     Returns the size of custom additional bond features vector associated with the molecules.
    #
    #     :return: The size of the additional bond feature vector.
    #     """
    #     return len(self._data[0].bond_features[0]) \
    #         if len(self._data) > 0 and self._data[0].bond_features is not None else None

    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0,
                           scale_atom_descriptors: bool = False, scale_bond_features: bool = False) -> StandardScaler:
        """
        Normalizes the features of the dataset using a :class:`~abtmpnn.data.StandardScaler`.

        The :class:`~abtmpnn.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each feature independently.

        If a :class:`~abtmpnn.data.StandardScaler` is provided, it is used to perform the normalization.
        Otherwise, a :class:`~abtmpnn.data.StandardScaler` is first fit to the features in this dataset
        and is then used to perform the normalization.

        :param scaler: A fitted :class:`~abtmpnn.data.StandardScaler`. If it is provided it is used,
                       otherwise a new :class:`~abtmpnn.data.StandardScaler` is first fitted to this
                       data and is then used.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        :param scale_atom_descriptors: If the features that need to be scaled are atom features rather than molecule.
        :param scale_bond_features: If the features that need to be scaled are bond descriptors rather than molecule.
        :return: A fitted :class:`~abtmpnn.data.StandardScaler`. If a :class:`~abtmpnn.data.StandardScaler`
                 is provided as a parameter, this is the same :class:`~abtmpnn.data.StandardScaler`. Otherwise,
                 this is a new :class:`~abtmpnn.data.StandardScaler` that has been fit on this dataset.
        """
        if len(self._data) == 0 or \
                (self._data[0].features is None and not scale_bond_features and not scale_atom_descriptors):
            return None

        if scaler is not None:
            self._scaler = scaler

        elif self._scaler is None:
            # if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
            #     features = np.vstack([d.raw_atom_descriptors for d in self._data])
            # elif scale_atom_descriptors and not self._data[0].atom_features is None:
            #     features = np.vstack([d.raw_atom_features for d in self._data])
            # elif scale_bond_features:
            #     features = np.vstack([d.raw_bond_features for d in self._data])
            # else:
            # stack all additional features for all molecules
            features = np.vstack([d.raw_features for d in self._data])
            self._scaler = StandardScaler(
                replace_nan_token=replace_nan_token)  # apply method
            self._scaler.fit(features)

        # if scale_atom_descriptors and not self._data[0].atom_descriptors is None:
        #     for d in self._data:
        #         d.set_atom_descriptors(self._scaler.transform(d.raw_atom_descriptors))
        # elif scale_atom_descriptors and not self._data[0].atom_features is None:
        #     for d in self._data:
        #         d.set_atom_features(self._scaler.transform(d.raw_atom_features))
        # elif scale_bond_features:
        #     for d in self._data:
        #         d.set_bond_features(self._scaler.transform(d.raw_bond_features))
        # else:
        for d in self._data:
            # the normalized features for all molecules
            d.set_features(self._scaler.transform(
                d.raw_features.reshape(1, -1))[0])

        return self._scaler

    def normalize_targets(self) -> StandardScaler:  # for regression tasks
        """
        Normalizes the targets of the dataset using a :class:`~abtmpnn.data.StandardScaler`.

        The :class:`~abtmpnn.data.StandardScaler` subtracts the mean and divides by the standard deviation
        for each task independently.

        This should only be used for regression datasets.

        :return: A :class:`~abtmpnn.data.StandardScaler` fitted to the targets.
        """
        targets = [d.raw_targets for d in self._data]
        scaler = StandardScaler().fit(targets)
        scaled_targets = scaler.transform(targets).tolist()
        self.set_targets(scaled_targets)

        return scaler

    def set_targets(self, targets: List[List[Optional[float]]]) -> None:
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats (or None) containing targets for each molecule. This must be the
                        same length as the underlying dataset.
        """
        assert len(self._data) == len(targets)
        for i in range(len(self._data)):
            self._data[i].set_targets(targets[i])

    def reset_features_and_targets(self) -> None:
        """Resets the features (atom, bond, and molecule) and targets to their raw values."""
        for d in self._data:
            d.reset_features_and_targets()

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e., the number of molecules).

        :return: The length of the dataset.
        """
        return len(self._data)

    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        r"""
        Gets one or more :class:`MoleculeDatapoint`\ s via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A :class:`MoleculeDatapoint` if an int is provided or a list of :class:`MoleculeDatapoint`\ s
                 if a slice is provided.
        """
        return self._data[item]


class MoleculeSampler(Sampler):
    """A :class:`MoleculeSampler` samples data from a :class:`MoleculeDataset` for a :class:`MoleculeDataLoader`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if :code:`shuffle` is True.
        """
        super(Sampler, self).__init__()

        self.dataset = dataset
        self.class_balance = class_balance
        self.shuffle = shuffle

        self._random = Random(seed)

        if self.class_balance:
            # find active molecules (if multi-task, then active means as long as there's a 1 in target vector
            indices = np.arange(len(dataset))
            #  then consider it as active)
            has_active = np.array(
                [any(target == 1 for target in datapoint.targets) for datapoint in dataset])

            self.positive_indices = indices[has_active].tolist()
            self.negative_indices = indices[~has_active].tolist()

            self.length = 2 * min(len(self.positive_indices),
                                  len(self.negative_indices))
        else:
            self.positive_indices = self.negative_indices = None

            self.length = len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        """Creates an iterator over indices to sample.
        __iter__ defines a method on a class which will return an iterator
        (an object that successively yields the next item contained by your object)."""
        if self.class_balance:
            if self.shuffle:
                self._random.shuffle(self.positive_indices)
                self._random.shuffle(self.negative_indices)

            indices = [index for pair in zip(
                self.positive_indices, self.negative_indices) for index in pair]
        else:
            indices = list(range(len(self.dataset)))

            if self.shuffle:
                self._random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of indices that will be sampled."""
        return self.length


def construct_molecule_batch(data: List[MoleculeDatapoint]) -> MoleculeDataset:
    r"""
    Constructs a :class:`MoleculeDataset` from a list of :class:`MoleculeDatapoint`\ s.

    Additionally, precomputes the :class:`~abtmpnn.features.BatchMolGraph` for the constructed
    :class:`MoleculeDataset`.

    :param data: A list of :class:`MoleculeDatapoint`\ s.
    :return: A :class:`MoleculeDataset` containing all the :class:`MoleculeDatapoint`\ s.
    """
    data = MoleculeDataset(data)
    data.batch_graph()  # Forces computation and caching of the BatchMolGraph for the molecules

    return data


class MoleculeDataLoader(DataLoader):
    """A :class:`MoleculeDataLoader` is a PyTorch :class:`DataLoader` for loading a :class:`MoleculeDataset`."""

    def __init__(self,
                 dataset: MoleculeDataset,
                 batch_size: int = 50,
                 num_workers: int = 8,
                 class_balance: bool = False,
                 shuffle: bool = False,
                 seed: int = 0):
        """
        :param dataset: The :class:`MoleculeDataset` containing the molecules to load.
        :param batch_size: Batch size.
        :param num_workers: Number of workers used to build batches.
        :param class_balance: Whether to perform class balancing (i.e., use an equal number of positive
                              and negative molecules). Class balance is only available for single task
                              classification datasets. Set shuffle to True in order to get a random
                              subset of the larger class.
        :param shuffle: Whether to shuffle the data.
        :param seed: Random seed. Only needed if shuffle is True.
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._class_balance = class_balance
        self._shuffle = shuffle
        self._seed = seed
        self._context = None
        self._timeout = 0
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread and self._num_workers > 0:
            self._context = 'forkserver'  # In order to prevent a hanging
            self._timeout = 3600  # Just for sure that the DataLoader won't hang

        self._sampler = MoleculeSampler(
            dataset=self._dataset,
            class_balance=self._class_balance,
            shuffle=self._shuffle,
            seed=self._seed
        )
        # inheritate from DataLoader
        super(MoleculeDataLoader, self).__init__(
            dataset=self._dataset,
            batch_size=self._batch_size,
            sampler=self._sampler,
            num_workers=self._num_workers,
            collate_fn=construct_molecule_batch,
            multiprocessing_context=self._context,
            timeout=self._timeout
        )

    @property
    def targets(self) -> List[List[Optional[float]]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats (or None) containing the targets.
        """
        if self._class_balance or self._shuffle:
            raise ValueError(
                'Cannot safely extract targets when class balance or shuffle are enabled.')

        return [self._dataset[index].targets for index in self._sampler]

    @property
    def iter_size(self) -> int:
        """Returns the number of data points included in each full iteration through the :class:`MoleculeDataLoader`."""
        return len(self._sampler)

    def __iter__(self) -> Iterator[MoleculeDataset]:
        r"""Creates an iterator which returns :class:`MoleculeDataset`\ s"""
        return super(MoleculeDataLoader, self).__iter__()
