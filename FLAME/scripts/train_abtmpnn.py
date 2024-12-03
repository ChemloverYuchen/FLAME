from FLAME import abtmpnn_train

if __name__ == '__main__':
    epoch = 50
    targets = ['abs', 'emi', 'plqy', 'e']
    for data_base in ['deep4chem', 'FluoDB']:
        for target in targets:
            train_data = f'data/{data_base}/{target}_train.csv'
            test_data = f'data/{data_base}/{target}_test.csv'
            valid_data = f'data/{data_base}/{target}_valid.csv'
            model_path = f'model/abtmpnn/{data_base}_{target}'
            abtmpnn_train(model_path, train_data, valid_data, test_data, epoch)
