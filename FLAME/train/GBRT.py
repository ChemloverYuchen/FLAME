import pandas as pd
import numpy as np
import joblib
import warnings
import os
from sklearn.ensemble import GradientBoostingRegressor
from FLAME.dataprocess.gbrt import get_GBRT_data

warnings.filterwarnings("ignore")

def gbrt_predict(model_path, output_file, input_file='', smiles=None):

    if os.path.exists(output_file):
        raise ValueError(f"out file: {output_file} already exist")

    if smiles==None:
        if len(input_file)==0:
            raise ValueError("At least One input: file or smiles")
        elif os.path.exists(input_file):
            data_df = pd.read_csv(input_file)
        else:
            raise ValueError(f"input data: {input_file} not exist")
    else:
        data_df = pd.DataFrame({'smiles':[x[0] for x in smiles], 'solvent':[x[1] for x in smiles]})
    
    df = get_GBRT_data(data_df, feature_save=False)

    clf = joblib.load(model_path)
    X_pre = np.array([ f'{r.sol},{r.CDK},{r.Ext},{r.Estat},{r.SF},{r.SFC}'.split(',') for i,r in df.iterrows()])
    df['pred'] = clf.predict(X_pre)
    
    for target in ['abs', 'emi', 'e', 'plqy', '']:
        if target in df.columns:
            break
    if len(target) ==0:
        df.loc[:, ['smiles', 'solvent', 'pred']].to_csv(output_file)
    else:
        df.loc[:, ['smiles', 'solvent', target, 'pred']].to_csv(output_file)
    return df['pred']

def gbrt_train(model_path, train_data, valid_data, test_data, epoch=None):
    if os.path.exists(model_path):
        raise ValueError(f"Model: {model_path} already exist")
    if not os.path.exists(train_data):
        raise ValueError(f"train Data: {train_data} not exist")
    if not os.path.exists(valid_data):
        raise ValueError(f"valid Data: {valid_data} not exist")
#     if not os.path.exists(test_data):
#         raise ValueError(f"test Data: {test_data} not exist")

    train_path = f'{train_data}_gbrt.csv'
    valid_path = f'{valid_data}_gbrt.csv'
    train_df = pd.read_csv(train_data)
    valid_df = pd.read_csv(valid_data)
    
    if 'SF' not in train_df.columns:
        train_df = get_GBRT_data(train_df, save_path=train_path, feature_save=False)
        
    if 'SF' not in valid_df.columns:
        valid_df = get_GBRT_data(valid_df, save_path=valid_path, feature_save=False)
        
    X_train = np.array([f'{r.sol},{r.CDK},{r.Ext},{r.Estat},{r.SF},{r.SFC}'.split(',') for i,r in train_df.iterrows()]).astype('float')
    for target in ['abs', 'emi', 'e', 'plqy', '']:
        if target in train_df.columns:
            break
    if len(target) ==0:
        raise ValueError('Data Do not contain any pred keyword')
    y_train = train_df[target]
    clf = GradientBoostingRegressor(learning_rate=0.05,
                                max_depth=31,
                                max_features=300,
                                min_samples_leaf=20,
                                n_estimators=1000,
                                verbose=1)
    print(f'Training model {target}')
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_path)
#     X_pre = np.array([ f'{r.sol},{r.CDK},{r.Ext},{r.Estat},{r.SF},{r.SFC}'.split(',') for i,r in valid_df.iterrows()])
#     valid_df['pred'] = clf.predict(X_pre)
#     print('Training model')