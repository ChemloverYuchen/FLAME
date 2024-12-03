# load data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from multiprocessing import Pool
from rdkit import Chem
import pickle
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys, AllChem
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

def smi2fp(data):
    smi, sol = data
    mol = Chem.MolFromSmiles(smi)
    maccs = MACCSkeys.GenMACCSKeys(mol).ToList()
    morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=33).ToList()
    smol = Chem.MolFromSmiles(sol)
    smaccs = MACCSkeys.GenMACCSKeys(smol).ToList()
    smorgan = AllChem.GetMorganFingerprintAsBitVect(smol, 2, nBits=33).ToList()
    return maccs+morgan+smaccs+smorgan

def smiles2inp(smiles, solvents, thd=10):
    worker = Pool(thd)
    fp_list = worker.map(smi2fp, zip(smiles, solvents))
    worker.close()
    worker.join()
    return np.array(fp_list)

def get_dataset(data_path, batch_size=32, shuffle=False):
    df = pd.read_csv(data_path)
    Y = df.iloc[:,-1].values
    X = smiles2inp(df['smiles'].values, df['solvent'].values)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).prefetch(-1)

def FCNN():
    model = keras.models.Sequential()
    l5 = keras.layers.Dense(512, activation='relu')
    l6 = keras.layers.Dropout(rate=0.2)
    l7 = keras.layers.Dense(128, activation='relu')
    l8 = keras.layers.Dense(30, activation='relu')
    l9 = keras.layers.Dense(1)
    layer = [l5, l6, l7, l8, l9]
    for i in range(len(layer)):
        model.add(layer[i])
    adam = keras.optimizers.Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='logcosh', metrics=['mae'])
    return model

def get_sch():
    def scheduler(epoch, lr):
        if epoch > 0 and epoch % 500 == 0:
            return lr * 0.1
        else:
            return lr
    return tf.keras.callbacks.LearningRateScheduler(scheduler)

def show_his(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="valid loss")
    plt.xlabel("Epochs", fontsize=16)
    plt.ylabel("LOSS", fontsize=16)
    plt.legend(fontsize=16)
    plt.show()
    
def fcnn_predict(model_path, output_file, input_file='', smiles=None):
    if os.path.exists(output_file):
        raise ValueError(f"out file: {output_file} already exist")
        
    if smiles==None:
        if len(input_file)==0:
            raise ValueError("At least One input: file or smiles")
        elif os.path.exists(input_file):
            df = pd.read_csv(input_file)
            x_input = smiles2inp(df.smiles, df.solvent)
        else:
            raise ValueError(f"input data: {input_file} not exist")
    else:
        df = pd.DataFrame({
            'smiles':[x[0] for x in smiles],
            'solvent':[x[1] for x in smiles]
        })
        x_input = smiles2inp(df.smiles, df.solvent)
    model = keras.models.load_model(model_path)
    y_pred = model.predict(x_input)
    df['pred'] = y_pred
    df.to_csv(output_file, index=False)
    
def fcnn_train(model_path, train_data, valid_data, test_data, epoch=2000):
    if os.path.exists(model_path):
        raise ValueError(f"Model: {model_path} already exist")
    if not os.path.exists(train_data):
        raise ValueError(f"train Data: {train_data} not exist")
    if not os.path.exists(valid_data):
        raise ValueError(f"valid Data: {valid_data} not exist")
    if not os.path.exists(test_data):
        raise ValueError(f"test Data: {test_data} not exist")
    train_dataset = get_dataset(train_data)
    valid_dataset = get_dataset(valid_data)
    test_dataset = get_dataset(test_data)
    model = FCNN()
    scheduler = get_sch()
    checkpoint = keras.callbacks.ModelCheckpoint(model_path, monitor='mae', verbose=1, save_best_only=True)
    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epoch,
        callbacks=[scheduler, checkpoint],
    )
    with open(f'{model_path}.pkl', 'wb') as file:
        pickle.dump(history, file)