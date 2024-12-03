from FLAME import uvvisml_predict
import os

if __name__ == '__main__':
    targets = ['abs', 'emi', 'plqy', 'e']
    for data_base in ['deep4chem', 'FluoDB']:
        for model in ['deep4chem', 'FluoDB']:
            for target in targets:
                model_path = f'model/uvvisml/{model}_{target}'
                input_file = f'data/{data_base}/{target}_test.csv'
                output_file = f'pred/{data_base}/uvvisml_{model}_{target}.csv'
                if not os.path.exists(f'pred/{data_base}/'):
                    os.makedirs(f'pred/{data_base}/')
                chemprop_predict(model_path, output_file, input_file=input_file)
