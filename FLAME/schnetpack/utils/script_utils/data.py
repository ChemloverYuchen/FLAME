import numpy as np
import torch
import os
from FLAME.dataprocess.schnet import train_test_split, AtomsLoader
from FLAME.dataprocess.schnet.ani1 import ANI1
from torch.utils.data.sampler import RandomSampler
from .script_error import ScriptError


__all__ = ["get_loaders", "get_statistics", "get_dataset"]


def get_statistics(
    args, split_path, train_loader, atomref, divide_by_atoms=False, logging=None
):
    """
    Get statistics for molecular properties. Use split file if possible.

    Args:
        args (argparse.Namespace): parsed script arguments
        split_path (str): path to the split file
        train_loader (spk.data.AtomsLoader): dataloader for training set
        atomref (dict): atomic references
        divide_by_atoms (dict or bool): divide mean by number of atoms if True
        logging: logger

    Returns:
        mean (dict): mean values for the selected properties
        stddev (dict): stddev values for the selected properties
    """
    # check if split file exists
    if not os.path.exists(split_path):
        raise ScriptError("No split file found ad {}".format(split_path))
    split_data = np.load(split_path)

    # check if split file contains statistical data
    if "mean" in split_data.keys():
        mean = {args.property: torch.from_numpy(split_data["mean"])}
        stddev = {args.property: torch.from_numpy(split_data["stddev"])}
        if logging is not None:
            logging.info("cached statistics was loaded...")

    # calculate statistical data
    else:
        mean, stddev = train_loader.get_statistics(
            args.property, divide_by_atoms, atomref
        )
        np.savez(
            split_path,
            train_idx=split_data["train_idx"],
            val_idx=split_data["val_idx"],
            test_idx=split_data["test_idx"],
            mean=mean[args.property].numpy(),
            stddev=stddev[args.property].numpy(),
        )

    return mean, stddev


def get_loaders(args, dataset, split_path, logging=None):
    """
    Create train-val-test splits for dataset and return the corresponding dataloaders.

    Args:
        args (argparse.Namespace): parsed script arguments
        dataset (spk.AtomsData): total dataset
        split_path (str): path to split file
        logging: logger

    Returns:
        (spk.AtomsLoader, spk.AtomsLoader, spk.AtomsLoader): dataloaders for train,
            val and test
    """
    if logging is not None:
        logging.info("create splits...")

    # create or load dataset splits depending on args.mode
    if args.mode == "train":
        data_train, data_val, data_test = train_test_split(
            dataset, *args.split, split_file=split_path
        )
    else:
        data_train, data_val, data_test = train_test_split(
            dataset, split_file=split_path
        )
        
    if logging is not None:
        logging.info("load data...")

    # build dataloaders
    train_loader = AtomsLoader(
        data_train,
        batch_size=args.batch_size,
        sampler=RandomSampler(data_train),
        num_workers=4,
        pin_memory=args.cuda,
    )
    val_loader = AtomsLoader(
        data_val, batch_size=args.batch_size, num_workers=2, pin_memory=args.cuda
    )
    test_loader = AtomsLoader(
        data_test, batch_size=args.batch_size, num_workers=2, pin_memory=args.cuda
    )

    return train_loader, val_loader, test_loader


def get_dataset(args, environment_provider, logging=None):
    """
    Get dataset from arguments.

    Args:
        args (argparse.Namespace): parsed arguments
        environment_provider (spk.environment.BaseEnvironmentProvider): environment-
            provider of dataset
        logging: logger

    Returns:
        spk.data.AtomsData: dataset

    """
    if args.dataset == "qm9":
        print('err')
        return None
    elif args.dataset == "ani1":
        if logging:
            logging.info("ANI1 will be loaded...")
        ani1 = ANI1(
            args.datapath,
            download=True,
            load_only=[args.property],
            collect_triples=args.model == "wacsf",
            num_heavy_atoms=args.num_heavy_atoms,
            environment_provider=environment_provider,
        )
        return ani1
    elif args.dataset == "custom":
        if logging:
            logging.info("Custom dataset will be loaded...")
        # define properties to be loaded
        load_only = [args.property]
        if args.derivative is not None:
            load_only.append(args.derivative)

        if args.stress is not None:
            load_only.append(args.stress)

        dataset = AtomsData(
            args.datapath,
            load_only=load_only,
            collect_triples=args.model == "wacsf",
            environment_provider=environment_provider,
        )
        return dataset
    else:
        raise ValueError("Invalid dataset selected!")
