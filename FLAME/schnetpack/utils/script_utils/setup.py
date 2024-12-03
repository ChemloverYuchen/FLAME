import os
import logging
from shutil import rmtree
from FLAME.schnetpack.utils.spk_utils import read_from_json, to_json, set_random_seed
from .script_error import ScriptError


__all__ = ["setup_run"]


def setup_run(args):
    argparse_dict = vars(args)
    jsonpath = os.path.join(args.modelpath, "args.json")
    if args.mode == "train":

        # build modeldir
        if args.overwrite and os.path.exists(args.modelpath):
            logging.info("existing model will be overwritten...")
            rmtree(args.modelpath)
        if not os.path.exists(args.modelpath):
            os.makedirs(args.modelpath)

        # store training arguments
        to_json(jsonpath, argparse_dict)

        set_random_seed(args.seed)
        train_args = args
    else:
        # check if modelpath is valid
        if not os.path.exists(args.modelpath):
            raise ScriptError(
                "The selected modeldir does not exist " "at {}!".format(args.modelpath)
            )

        # load training arguments
        train_args = read_from_json(jsonpath)

    # apply alias definitions
    train_args = apply_aliases(train_args)
    return train_args


def apply_aliases(args):
    # force alias for custom dataset
    if args.dataset == "custom":
        if args.force is not None:
            if args.derivative is not None:
                raise ScriptError(
                    "Force and derivative define the same property. Please don`t use "
                    "both."
                )
            args.derivative = args.force
            args.negative_dr = True

            # add rho value if selected
            if "force" in args.rho.keys():
                args.rho["derivative"] = args.rho.pop("force")

    return args
