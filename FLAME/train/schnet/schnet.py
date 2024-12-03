import numpy as np
import torch
from .metrics import MeanAbsoluteError, RootMeanSquaredError
from FLAME.dataprocess.schnet import AtomsDataSubset, AtomsLoader, ANI1
from FLAME.schnetpack.utils import (
    get_model,
    get_trainer
)

import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from FLAME.utils import get_schnet_data

class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = DotDict(value)
        return value

def _dataset(db_path):
    print('load ', db_path)
#     environment_provider = spk.environment.SimpleEnvironmentProvider()
    dataset = ANI1(
                db_path,
                download=False,
                load_only=['energy'],
                collect_triples=False,
                num_heavy_atoms=8
    )
    dataset = AtomsDataSubset(dataset, np.random.permutation(np.arange(0, len(dataset))).tolist())
    loader = AtomsLoader(
        dataset,
        batch_size=100,
        sampler=RandomSampler(dataset),
        num_workers=4,
        pin_memory=False,
    )
    return loader

def schnet_train(model_path, train_data, valid_data, test_data, epoch):
    if os.path.exists(model_path):
        raise ValueError(f"Model: {model_path} already exist")
    
    if train_data[-2:] != 'db':
        train_data = train_data+'.db'
    if valid_data[-2:] != 'db':
        valid_data = valid_data+'.db'

    if not os.path.exists(train_data):
        raise ValueError(f"train Data: {train_data} not exist, Please get_sechnet_data first")
    if not os.path.exists(valid_data):
        raise ValueError(f"valid Data: {valid_data} not exist, Please get_sechnet_data first")
    if not os.path.exists(test_data):
        raise ValueError(f"test Data: {test_data} not exist")

    train_loader = _dataset(train_data)
    valid_loader = _dataset(valid_data)
    fargs = {'mode': 'train',
     'model': 'schnet',
     'modelpath': model_path,
     'dataset':'ani1',
     'cuda': False,
     'parallel': False,
     'seed': None,
     'overwrite': False,
     'split_path': None,
     'max_epochs': 1000,
     'max_steps': None,
     'lr': 0.0001,
     'lr_patience': 25,
     'lr_decay': 0.8,
     'lr_min': 1e-06,
     'logger': 'csv',
     'log_every_n_epochs': 1,
     'n_epochs': 100,
     'checkpoint_interval': 1,
     'keep_n_checkpoints': 3,
     'environment_provider_device': 'cpu',
     'features': 128,
     'interactions': 6,
     'cutoff_function': 'cosine',
     'num_gaussians': 50,
     'normalize_filter': False,
     'property': 'energy',
     'cutoff': 10.0,
     'batch_size': 100,
     'environment_provider': 'simple',
     'num_heavy_atoms': 8}
    fargs = DotDict(fargs)
    
    device = torch.device("cpu")
    metrics = [
        MeanAbsoluteError('energy', 'energy'),
        RootMeanSquaredError('energy', 'energy'),
    ]
    mean, stddev = valid_loader.get_statistics('energy', True)
    model = get_model(fargs, train_loader, mean, stddev, {'energy': None})
    trainer = get_trainer(fargs, model, train_loader, valid_loader, metrics)
    trainer.train(device, n_epochs=epoch)

def get_test_dataset(data_path):
#     environment_provider = spk.environment.SimpleEnvironmentProvider()
    dataset = ANI1(
                data_path,
                download=False,
                load_only=['energy', 'indices'],
                collect_triples=False,
                num_heavy_atoms=8
    )
    dataset = AtomsDataSubset(dataset, np.random.permutation(np.arange(0, len(dataset))).tolist())
    loader = AtomsLoader(
        dataset,
        batch_size=100,
#         sampler=RandomSampler(dataset),
        sampler=SequentialSampler(dataset),
        num_workers=4,
        pin_memory=False,
    )
    return loader

def schnet_predict(model_path, output_file, input_file='', smiles=[]):
    best_model = torch.load(model_path+'/best_model')
    if len(smiles)==0:
        if len(input_file)==0:
            raise ValueError("At least One input: file or smiles")
        elif os.path.exists(input_file):
            if os.path.exists(output_file):
                raise ValueError(f"out file: {output_file} already exist")
            
            if input_file[-2:] != 'db':
                df = pd.read_csv(input_file)
                data_path = f'{input_file}.db'
                get_schnet_data(df.iloc[:,0].values, np.zeros(len(smiles)), df.index.values, save_path=data_path, conf_num=1)
                test_loader = get_test_dataset(data_path)
            else:
                test_loader = get_test_dataset(input_file)
            pred = np.array([])
            indices = np.array([])
            for batch in test_loader:
                result = best_model(batch)['energy'].detach().numpy()
                pred = np.concatenate([pred, result.reshape(-1)])
                indices = np.concatenate([indices, batch['indices'].detach().numpy().reshape(-1)])
        else:
            raise ValueError(f"input data: {input_file} not exist")
    else:
        df = pd.DataFrame({
            'smiles':smiles,
        })
        data_path = 'schnet_tmp.db'
        get_schnet_data(smiles, np.zeros(len(smiles)), df.index.values, save_path=data_path, conf_num=1)
        test_loader = get_test_dataset(data_path)
        pred = np.array([])
        indices = np.array([])
        for batch in test_loader:
            result = best_model(batch)['energy'].detach().numpy()
            pred = np.concatenate([pred, result.reshape(-1)])
            indices = np.concatenate([indices, batch['indices'].detach().numpy().reshape(-1)])
    
    # if model_path.split('/')[-2].find('abs') > -1 or model_path.split('/')[-2].find('emi') > -1:
    #     pred = 1240./np.array(pred)
    df.loc[indices, 'pred'] = pred
    df.to_csv(output_file, index=False)
    if len(pred)!=len(df):
        print(f'Shape mismatch please check outputfile {output_file}')
        return None
    else:        
        return pred