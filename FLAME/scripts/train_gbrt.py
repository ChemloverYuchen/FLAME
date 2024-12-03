from FLAME import gbrt_train
import os

if __name__ == '__main__':
    targets = ['abs', 'emi', 'plqy', 'e']
    for data_base in ['deep4chem', 'FluoDB']:
        for target in targets:
            train_data = f'data/{data_base}/{target}_train.csv'
            test_data = f'data/{data_base}/{target}_test.csv'
            valid_data = f'data/{data_base}/{target}_valid.csv'
            if not os.path.exists('model/gbrt'):
                os.makedirs('model/gbrt')
            model_path = f'model/gbrt/{database}_{target}.m'
            gbrt_train(model_path, train_data, valid_data, test_data)
