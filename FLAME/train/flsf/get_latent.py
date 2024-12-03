from collections import OrderedDict
import csv
import os
from typing import List, Optional, Union

import pickle

import numpy as np
from tqdm import tqdm

from FLAME.flsf.args import PredictArgs, TrainArgs
from FLAME.dataprocess.flsf.data import get_data, get_data_from_smiles, MoleculeDataLoader, MoleculeDataset, StandardScaler
from FLAME.flsf.utils import load_args, load_checkpoint, load_scalers, makedirs, timeit, update_prediction_args
from FLAME.dataprocess.flsf.features import set_extra_atom_fdim, set_extra_bond_fdim

from typing import List

import torch
from FLAME.flsf.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~flsf.models.model.MoleculeModel`.
    :param data_loader: A :class:`~flsf.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~flsf.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()
    features_in_hook = []
    features_out_hook = []
    
    def hook(module, fea_in, fea_out):
        features_in_hook.append(fea_in)
        features_out_hook.append(fea_out)
        return None

#     layer_name = ['encoder.gru', 'encoder.encoder.0.act_func', 'encoder.encoder.1.act_func']
    layer_name = ['encoder.gru', 'ffn']
    for (name, module) in model.named_modules():
        if name in layer_name:
            module.register_forward_hook(hook=hook)

    preds = []
    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, mol_adj_batch, mol_dist_batch, mol_clb_batch = \
            batch.batch_graph(), batch.features(), batch.adj_features(
            ), batch.dist_features(), batch.clb_features()

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, features_batch,
                                mol_adj_batch, mol_dist_batch, mol_clb_batch)

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)
    
#     features = []
#     for f in features_out_hook:
#         features.append([x.data.cpu().numpy() for x in f])
#     features = np.vstack([x[1][0].numpy() for x in features_in_hook])
    gru_latent = features_in_hook[::2]
    all_latent = features_in_hook[1::2]
    
    gru_feature = np.vstack([x[1][0].numpy() for x in gru_latent])
    all_feature = np.vstack([x[0].numpy() for x in all_latent])
    features = [gru_feature, all_feature[:,:1900], all_feature[:,1900:]]
    return preds, features


@timeit()
def make_predictions(args: PredictArgs, smiles: List[List[str]] = None, feature_file='feature.pkl') -> List[List[Optional[float]]]:
    """
    Loads data and a trained model and uses the model to make predictions on the data.

    If SMILES are provided, then makes predictions on smiles.
    Otherwise makes predictions on :code:`args.test_data`.

    :param args: A :class:`~flsf.args.PredictArgs` object containing arguments for
                 loading data and a model and making predictions.
    :param smiles: List of list of SMILES to make predictions on.
    :return: A list of lists of target predictions.
    """
    print('Loading training args')
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    update_prediction_args(predict_args=args, train_args=train_args)
    args: Union[PredictArgs, TrainArgs]

    if args.atom_descriptors == 'feature':
        set_extra_atom_fdim(train_args.atom_features_size)

    if args.bond_features_path is not None:
        set_extra_bond_fdim(train_args.bond_features_size)

    print('Loading data')
    if smiles is not None:
        full_data = get_data_from_smiles(
            smiles=smiles,
            skip_invalid_smiles=False,
            features_generator=args.features_generator
        )
    else:
        full_data = get_data(path=args.test_path, smiles_columns=args.smiles_columns, target_columns=[], ignore_columns=[],
                             skip_invalid_smiles=False, args=args, store_row=not args.drop_extra_columns)

    print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if all(mol is not None for mol in full_data[full_index].mol):
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    print(f'Test size = {len(test_data):,}')

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Partial results for variance robust calculation.
    if args.ensemble_variance:
        all_preds = np.zeros((len(test_data), num_tasks, len(args.checkpoint_paths)))

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    
    for index, checkpoint_path in enumerate(tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths))):
        # Load model and scalers
        model = load_checkpoint(checkpoint_path, device=args.device)
        scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler = load_scalers(checkpoint_path)

        # Normalize features
        if args.features_scaling or train_args.atom_descriptor_scaling or train_args.bond_feature_scaling:
            test_data.reset_features_and_targets()
            if args.features_scaling:
                test_data.normalize_features(features_scaler)
            if train_args.atom_descriptor_scaling and args.atom_descriptors is not None:
                test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
            if train_args.bond_feature_scaling and args.bond_features_size > 0:
                test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)

        # Make predictions
        model_preds, feature = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler
        )
        pickle.dump(feature, open(feature_file, 'wb'))
        sum_preds += np.array(model_preds)
        if args.ensemble_variance:
            all_preds[:, :, index] = model_preds

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)
    avg_preds = avg_preds.tolist()

    if args.ensemble_variance:
        all_epi_uncs = np.var(all_preds, axis=2)
        all_epi_uncs = all_epi_uncs.tolist()

    # Save predictions
    print(f'Saving predictions to {args.preds_path}')
    assert len(test_data) == len(avg_preds)
    if args.ensemble_variance:
        assert len(test_data) == len(all_epi_uncs)
    makedirs(args.preds_path, isfile=True)

    # Get prediction column names
    if args.dataset_type == 'multiclass':
        task_names = [f'{name}_class_{i}' for name in task_names for i in range(args.multiclass_num_classes)]
    else:
        task_names = task_names
    
    # Copy predictions over to full_data
    for full_index, datapoint in enumerate(full_data):
        valid_index = full_to_valid_indices.get(full_index, None)
        preds = avg_preds[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)
        if args.ensemble_variance:
            epi_uncs = all_epi_uncs[valid_index] if valid_index is not None else ['Invalid SMILES'] * len(task_names)

        # If extra columns have been dropped, add back in SMILES columns
        if args.drop_extra_columns:
            datapoint.row = OrderedDict()

            smiles_columns = args.smiles_columns

            for column, smiles in zip(smiles_columns, datapoint.smiles):
                datapoint.row[column] = smiles

        # Add predictions columns
        if args.ensemble_variance:
            for pred_name, pred, epi_unc in zip(task_names, preds, epi_uncs):
                datapoint.row[pred_name] = pred
                datapoint.row[pred_name+'_epi_unc'] = epi_unc
        else:
            for pred_name, pred in zip(task_names, preds):
                datapoint.row[pred_name] = pred

    # Save
    with open(args.preds_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=full_data[0].row.keys())
        writer.writeheader()

        for datapoint in full_data:
            writer.writerow(datapoint.row)

    return avg_preds


def get_flsf_latent(model_path, output_file, input_file='', smiles=None) -> None:
    """Parses Chemprop predicting arguments and runs prediction using a trained Chemprop model.

    This is the entry point for the command line command :code:`flsf_predict`.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model: {model_path} not exist")
    # if os.path.exists(output_file):
    #     raise ValueError(f"output file: {output_file} already exist")

    pred_args = ['--test_path', input_file, '--checkpoint_dir', model_path,
            '--preds_path', output_file+'.csv','--number_of_molecules', '2', '--no_cuda']

    if smiles==None:
        if len(input_file)==0:
            raise ValueError("At least One input: file or smiles")
        elif os.path.exists(input_file):
            make_predictions(args=PredictArgs().parse_args(pred_args), feature_file=output_file+'.pkl')
        else:
            raise ValueError(f"input data: {input_file} not exist")
    else:
        make_predictions(args=PredictArgs().parse_args(pred_args), smiles=smiles)
