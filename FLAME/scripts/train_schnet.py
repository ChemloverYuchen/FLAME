from FLAME import schnet_train
import os

if __name__ == '__main__':
    epoch = 100
    targets = ['abs', 'emi', 'plqy', 'e']
    for data_base in ['deep4chem', 'FluoDB']:
        for target in targets:
            train_data = f'data/schnet/{data_base}/{target}_train.db'
            test_data = f'data/schnet/{data_base}/{target}_test.db'
            valid_data = f'data/schnet/{data_base}/{target}_valid.db'
            if not os.path.exists('model/schnet'):
                os.makedirs('model/schnet')
            model_path = f'model/schnet/{data_base}_{target}'
            schnet_train(model_path, train_data, valid_data, test_data, epoch)
