from FLAME import fcnn_train
import os

if __name__ == '__main__':
    epoch = 2000
    targets = ['abs', 'emi', 'plqy', 'e']
    for data_base in ['deep4chem', 'FluoDB']:
        for target in targets:
            train_data = f'data/{data_base}/{target}_train.csv'
            test_data = f'data/{data_base}/{target}_test.csv'
            valid_data = f'data/{data_base}/{target}_valid.csv'
            if os.path.exists('model/fcnn'):
                os.makedirs('model/fcnn')
            model_path = f'model/fcnn/{data_base}_{target}.h5'
            fcnn_train(model_path, train_data, valid_data, test_data, epoch)
