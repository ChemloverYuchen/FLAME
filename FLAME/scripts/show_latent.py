from FLAME import get_flsf_latent
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# from FLAME.flsf.utils import load_checkpoint
# model = load_checkpoint('FLAME/model/flsf_scaffold/FluoDB_abs/fold_0/model_0/model.pt')

def draw(feature, title):
    pca = PCA(n_components=2)
    pca.fit(feature)
    embedding = pca.fit_transform(feature)
    plt.figure(figsize=(9, 6))
    hb=plt.scatter(embedding[:,0], embedding[:,1], s=1,c=real,cmap="rainbow",marker='o')
    cb=plt.colorbar(hb)
    cb.ax.tick_params(labelsize=12, width=1.2)
    cb.outline.set_linewidth(1.2)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim([-4,4])
    plt.ylim([-4,4])
    ax=plt.gca()
    ax.spines[:].set_linewidth(1.2)
    ax.tick_params(labelsize=12,width=1.2)
    # plt.savefig('plot/test.svg', dpi=600, format='svg', bbox_inches='tight')
    if not os.path.exists('plot'):
        os.makedirs('plot')
    plt.savefig(f'plot/{title}.png', dpi=300, format='png', bbox_inches='tight')

if __name__ == '__main__':
    target = 'abs'
    tag = '_scaffold'
    model = 'FluoDB'

    input_file = f'data/FluoDB/{target}_test.csv'
    output_file = 'flsf_latent'
    model_path = f'model/flsf{tag}/{model}_{target}'
    df = pd.read_csv(input_file)
    get_flsf_latent(model_path, output_file, input_file=input_file)

    features = pickle.load(open('flsf_latent.pkl', 'rb'))
    mol_feature = features[0]
    tag_feature = features[1]
    sol_feature = features[2]
    mix_feature = np.hstack([tag_feature, sol_feature])
    if target == 'e':
        df[target] = np.log10(df[target])
    real = df[target].values
    draw(mol_feature, f'mol_{target}{tag}')
    draw(tag_feature, f'tag_{target}{tag}')
    draw(mix_feature, f'mix_{target}{tag}')