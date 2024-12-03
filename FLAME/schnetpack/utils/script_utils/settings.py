# import FLAME.schnetpack as spk
from FLAME.dataprocess.schnet.ani1 import ANI1
from FLAME.schnetpack.environment import SimpleEnvironmentProvider, AseEnvironmentProvider, TorchEnvironmentProvider

__all__ = [
    "divide_by_atoms",
    "pooling_mode",
    "get_divide_by_atoms",
    "get_pooling_mode",
    "get_negative_dr",
    "get_derivative",
    "get_contributions",
    "get_stress",
    "get_module_str",
    "get_environment_provider",
]


divide_by_atoms = {
    ANI1.energy: True,
}

pooling_mode = {
    ANI1.energy: "sum",
}


def get_divide_by_atoms(args):
    """
    Get 'divide_by_atoms'-parameter depending on run arguments.
    """
    if args.dataset == "custom":
        return args.aggregation_mode == "sum"
    return divide_by_atoms[args.property]


def get_pooling_mode(args):
    """
    Get 'pooling_mode'-parameter depending on run arguments.
    """
    if args.dataset == "custom":
        return args.aggregation_mode
    return pooling_mode[args.property]


def get_derivative(args):
    if args.dataset == "custom":
        return args.derivative
    # elif args.dataset == "md17" and not args.ignore_forces:
    #     return spk.datasets.MD17.forces
    return None


def get_contributions(args):
    if args.dataset == "custom":
        return args.contributions
    return None


def get_stress(args):
    if args.dataset == "custom":
        return args.stress
    return None


def get_negative_dr(args):
    if args.dataset == "custom":
        return args.negative_dr
    elif args.dataset == "md17":
        return True
    return False


def get_module_str(args):
    if args.dataset == "custom":
        return args.output_module
    if args.model == "schnet":
        # if args.property == spk.datasets.QM9.mu:
        #     return "dipole_moment"
        # if args.property == spk.datasets.QM9.r2:
        #     return "electronic_spatial_extent"
        return "atomwise"
    elif args.model == "wacsf":
        # if args.property == spk.datasets.QM9.mu:
        #     return "elemental_dipole_moment"
        return "elemental_atomwise"


def get_environment_provider(args, device):
    if args.environment_provider == "simple":
        return SimpleEnvironmentProvider()
    elif args.environment_provider == "ase":
        return AseEnvironmentProvider(cutoff=args.cutoff)
    elif args.environment_provider == "torch":
        return TorchEnvironmentProvider(
            cutoff=args.cutoff, device=device
        )
    else:
        raise NotImplementedError
