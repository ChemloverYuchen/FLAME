from FLAME import flsf_train
# tag='_maccs'
if __name__ == '__main__':
    epoch = 50
    targets = ['abs', 'emi', 'plqy', 'e']
    for data_base in ['deep4chem', 'FluoDB']:
        for target in targets:
            train_data = f'data/{data_base}/{target}_train.csv'
            test_data = f'data/{data_base}/{target}_test.csv'
            valid_data = f'data/{data_base}/{target}_valid.csv'
            model_path = f'model/flsf/{data_base}_{target}'
            flsf_train(model_path, train_data, valid_data, test_data, epoch)