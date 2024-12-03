import torch
import logging
import FLAME.schnetpack.nn as snn
from FLAME.schnetpack.representation.schnet import SchNet
from FLAME.schnetpack.representation.hdnn import StandardizeSF,BehlerSFBlock
from FLAME.schnetpack import representation
from FLAME.schnetpack.utils.spk_utils import count_params
from .script_error import ScriptError
from .settings import  get_module_str,get_pooling_mode, get_derivative, get_negative_dr, get_contributions, get_stress
from ase.data import atomic_numbers
from .output_modules import *
import torch.nn as nn

__all__ = ["get_representation", "get_output_module", "get_model", "AtomisticModel"]

from torch import nn as nn
import torch

from FLAME.schnetpack import Properties

class ModelError(Exception):
    pass

class AtomisticModel(nn.Module):
    """
    Join a representation model with output modules.

    Args:
        representation (torch.nn.Module): Representation block of the model.
        output_modules (list or nn.ModuleList or spk.output_modules.Atomwise): Output
            block of the model. Needed for predicting properties.

    Returns:
         dict: property predictions
    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()
        self.representation = representation
        if type(output_modules) not in [list, nn.ModuleList]:
            output_modules = [output_modules]
        if type(output_modules) == list:
            output_modules = nn.ModuleList(output_modules)
        self.output_modules = output_modules
        # For gradients
        self.requires_dr = any([om.derivative for om in self.output_modules])
        # For stress tensor
        self.requires_stress = any([om.stress for om in self.output_modules])

    def forward(self, inputs):
        """
        Forward representation output through output modules.
        """
        if self.requires_dr:
            inputs[Properties.R].requires_grad_()
        if self.requires_stress:
            # Check if cell is present
            if inputs[Properties.cell] is None:
                raise ModelError("No cell found for stress computation.")

            # Generate Cartesian displacement tensor
            displacement = torch.zeros_like(inputs[Properties.cell]).to(
                inputs[Properties.R].device
            )
            displacement.requires_grad = True
            inputs["displacement"] = displacement

            # Apply to coordinates and cell
            inputs[Properties.R] = inputs[Properties.R] + torch.matmul(
                inputs[Properties.R], displacement
            )
            inputs[Properties.cell] = inputs[Properties.cell] + torch.matmul(
                inputs[Properties.cell], displacement
            )

        inputs["representation"] = self.representation(inputs)
        outs = {}
        for output_model in self.output_modules:
            outs.update(output_model(inputs))
        return outs



def get_representation(args, train_loader=None):
    # build representation
    if args.model == "schnet":

        cutoff_network = snn.cutoff.get_cutoff_by_string(args.cutoff_function)

        return SchNet(
            n_atom_basis=args.features,
            n_filters=args.features,
            n_interactions=args.interactions,
            cutoff=args.cutoff,
            n_gaussians=args.num_gaussians,
            cutoff_network=cutoff_network,
            normalize_filter=args.normalize_filter,
        )

    elif args.model == "wacsf":
        sfmode = ("weighted", "Behler")[args.behler]
        # Convert element strings to atomic charges
        elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
        representation = BehlerSFBlock(
            args.radial,
            args.angular,
            zetas=set(args.zetas),
            cutoff_radius=args.cutoff,
            centered=args.centered,
            crossterms=args.crossterms,
            elements=elements,
            mode=sfmode,
        )
        logging.info(
            "Using {:d} {:s}-type SF".format(representation.n_symfuncs, sfmode)
        )
        # Standardize representation if requested
        if args.standardize:
            if train_loader is None:
                raise ValueError(
                    "Specification of a training_loader is required to standardize "
                    "wACSF"
                )
            else:
                logging.info("Computing and standardizing symmetry function statistics")
                return StandardizeSF(
                    representation, train_loader, cuda=args.cuda
                )

        else:
            return representation

    else:
        raise NotImplementedError("Unknown model class:", args.model)


def get_output_module_by_str(module_str):
    if module_str == "atomwise":
        return Atomwise
    elif module_str == "elemental_atomwise":
        return ElementalAtomwise
    elif module_str == "dipole_moment":
        return DipoleMoment
    elif module_str == "elemental_dipole_moment":
        return ElementalDipoleMoment
    elif module_str == "polarizability":
        return Polarizability
    elif module_str == "electronic_spatial_sxtent":
        return ElectronicSpatialExtent
    else:
        raise ScriptError(
            "{} is not a valid output " "module!".format(module_str)
        )


def get_output_module(args, representation, mean, stddev, atomref):
    derivative = get_derivative(args)
    negative_dr = get_negative_dr(args)
    contributions = get_contributions(args)
    stress = get_stress(args)
    if args.dataset == "md17" and not args.ignore_forces:
        print('error')
        return None
    output_module_str = get_module_str(args)
    if output_module_str == "dipole_moment":
        return DipoleMoment(
            args.features,
            predict_magnitude=True,
            mean=mean[args.property],
            stddev=stddev[args.property],
            property=args.property,
            contributions=contributions,
        )
    elif output_module_str == "electronic_spatial_extent":
        return ElectronicSpatialExtent(
            args.features,
            mean=mean[args.property],
            stddev=stddev[args.property],
            property=args.property,
            contributions=contributions,
        )
    elif output_module_str == "atomwise":
        return Atomwise(
            args.features,
            aggregation_mode=get_pooling_mode(args),
            mean=mean[args.property],
            stddev=stddev[args.property],
            atomref=atomref[args.property],
            property=args.property,
            derivative=derivative,
            negative_dr=negative_dr,
            contributions=contributions,
            stress=stress,
        )
    elif output_module_str == "polarizability":
        return Polarizability(
            args.features,
            aggregation_mode=get_pooling_mode(args),
            property=args.property,
        )
    elif output_module_str == "isotropic_polarizability":
        return Polarizability(
            args.features,
            aggregation_mode=get_pooling_mode(args),
            property=args.property,
            isotropic=True,
        )
    # wacsf modules
    elif output_module_str == "elemental_dipole_moment":
        elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
        return ElementalDipoleMoment(
            representation.n_symfuncs,
            n_hidden=args.n_nodes,
            n_layers=args.n_layers,
            predict_magnitude=True,
            elements=elements,
            property=args.property,
        )
    elif output_module_str == "elemental_atomwise":
        elements = frozenset((atomic_numbers[i] for i in sorted(args.elements)))
        return ElementalAtomwise(
            representation.n_symfuncs,
            n_hidden=args.n_nodes,
            n_layers=args.n_layers,
            aggregation_mode=get_pooling_mode(args),
            mean=mean[args.property],
            stddev=stddev[args.property],
            atomref=atomref[args.property],
            elements=elements,
            property=args.property,
            derivative=derivative,
            negative_dr=negative_dr,
        )
    else:
        raise NotImplementedError


def get_model(args, train_loader, mean, stddev, atomref, logging=None):
    """
    Build a model from selected parameters or load trained model for evaluation.

    Args:
        args (argsparse.Namespace): Script arguments
        train_loader (spk.AtomsLoader): loader for training data
        mean (torch.Tensor): mean of training data
        stddev (torch.Tensor): stddev of training data
        atomref (dict): atomic references
        logging: logger

    Returns:
        spk.AtomisticModel: model for training or evaluation
    """
    if args.mode == "train":
        if logging:
            logging.info("building model...")
        representation = get_representation(args, train_loader)
        output_module = get_output_module(
            args,
            representation=representation,
            mean=mean,
            stddev=stddev,
            atomref=atomref,
        )
        model = AtomisticModel(representation, [output_module])

        if args.parallel:
            model = nn.DataParallel(model)
        if logging:
            logging.info(
                "The model you built has: %d parameters" % count_params(model)
            )
        return model
    else:
        raise ValueError("Invalid mode selected: {}".format(args.mode))
