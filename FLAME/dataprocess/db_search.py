import os
import pandas as pd
from collections import Counter
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from tqdm import tqdm
from functools import partial

def is_valid(smi):
    if len(smi) < 2:
        return False
    try:
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            return False
        Chem.Kekulize(mol)
    except:
        return False
    return True

def smi2fp64(fp_type, smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        # fp = Chem.RDKFingerprint(mol)
            return fp.ToBitString()
        # return fp.ToBase64()
    except:
        return ''

def run_search(smi, sim_limit=0.6, fp_type='morgan', output_file=''):
    if type(smi) != str:
        raise ValueError('Please input a single SMILES string.')
    if not is_valid(smi):
        raise ValueError(f"Invalid SMILES input: {smi}")
    pwd = os.path.dirname(os.path.normpath(__file__))
    get_fp = partial(smi2fp64, fp_type)
    afp = get_fp(smi)
    afp = DataStructs.CreateFromBitString(afp)
    h5_db = f'{pwd}/FluoDB_{fp_type}.h5'
    csv_db = f'{pwd}/FluoDB.csv'
    print('Loading DB')
    if os.path.exists(h5_db):
        df = pd.read_hdf(h5_db, key='df')
    elif os.path.exists(csv_db):
        print(f'Process {fp_type} DB (Only at first running)')
        df = pd.read_csv(csv_db)
        df['fps'] = list(map(get_fp, df['smiles']))
        df.to_hdf(h5_db, complevel=5, key='df', mode='w', format='table')
    else:
        raise ValueError('DB file not found!')
    print('Start Searching')
    sims = []
    df = df[df['fps'].str.len()>0]
    for fp in tqdm(df['fps']):
        if len(fp) == 0:
            sim = 0
        else:
            sfp = DataStructs.CreateFromBitString(fp)
            sim = DataStructs.TanimotoSimilarity(sfp, afp)
        sims.append(sim)
    print('Searching Done!')
    df['sim'] = sims
    df_res = df[df.sim>sim_limit].drop(columns=['fps']).sort_values(by='sim',ascending=False).reset_index(drop=True)
    print(f'Find {len(df_res)} molecules similarity greater than {sim_limit}.')
    # targets = Counter(df_res["Target Name Assigned by Curator or DataSource"].values)
    # target = targets.most_common(3)
    if len(output_file) > 0:
        print(f'Result is saved in {output_file}')
        df_res.to_csv(output_file, index=False)
    return df_res